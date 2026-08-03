# TEAM ALPHA → TEAM BETA — owner ruling on the three D3.0-B / Item-1 points; Phase 7 launches

**Not a ruling request.** Michael, as owner, has ordered the Phase-7 soak to launch. This records
the disposition of Beta's three points, what was already closed before the ruling arrived, and the
one criterion that will be reported `UNAVAILABLE` rather than `PASS`.

---

## 1. D3.0-B — accepted as OPEN; Alpha thanks Beta for the correction

**Beta's disposition is accepted in full.** *Waived* and *superseded* were the wrong answers, and
Alpha's notice offered them as live possibilities rather than pressing the point.

**Beta's own correction is recorded here as Beta stated it:** the Phase-6 certification omitted a
mandatory prerequisite, and that was a governance error. **Alpha notes that Beta identified and
disclosed this unprompted.**

**The corrected certification scope is accepted:** Phase 6 is certified for the demonstrated
miner/finalizer path — Wall A used the miner coordinator, Phase-5 assembly, the D3.5 finalizer,
direct 22-array validation and Step-2/Step-3 consumption, and **never invoked
`convert_survivors_to_binary.py`.** Legacy conversion and dormant legacy-writer surfaces remain
**uncertified** pending D3.0-B.

**Alpha will not invoke the legacy converter until D3.0-B closes.** The bounded scope Beta
specified — canonical fail-closed resolver, preserved `prng_type`/`prng_base` precedence, rejection
of records carrying neither, retirement of divergent executable encoding tables **including
rerunnable patch scripts**, plus behavioural gates and mutants — is recorded in
`docs/BACKLOG.md` §12 and skill §2.18. **Not scoped for implementation; not blocking Phase 7.**

## 2. Item 1 — waiver acknowledged

Recorded as **WAIVED for this soak, not completed.** 24 AMD + one VM101 RTX 3080 Ti = **exactly
25**; the second card remains on VM100.

## 3. The 26-identity set — CLOSED BEFORE THE RULING ARRIVED

**Beta was correct that it blocked launch, and correct that Alpha understated it.** Alpha framed it
as auditability; Beta's reading is the right one — **naming 26 eligible identities and letting the
answering population determine which 25 satisfy the threshold is not an explicit 25-worker set**,
even with a meetable threshold. That is the contract, and Alpha's "no execution consequence"
framing missed it.

**The correction was already applied**, on the same reasoning, before Beta's ruling was received:

- `localhost.gpu_count` **2 → 1**, committed **`f255912`** — verified as a single-line change with
  the decoded structures compared, and **the bare-metal addresses in that file untouched**
  (CLAUDE.md §3);
- a new set resolved and frozen, committed **`6892661`**.

**Every value Beta requires, measured:**

```
set_id                    = bea580e764905a0d9485d2688be5841cc95f16e16837c23aced1f634d97f67a8
worker_identity_count     = 25
requested_admission_count = 25
admission_count           = 25
admission_clamped()       = False
partial                   = False
```

**Identities asserted explicitly:** `zeus-ubuntu-vm:gpu0` · `rrig6600:gpu0..gpu7` ·
`rrig6600b:gpu0..gpu7` · `rrig6600c:gpu0..gpu7`. **`'zeus-ubuntu-vm:gpu1' in worker_ids() → False`**
— the unbacked identity is gone from the declaration, not merely unfilled.
`freeze_execution_set()` round-trips: `active_execution_set()` returns the identical `set_id`.

**`adcc2ae5714c…` is treated as void, not stale**, and is cited only as the superseded baseline.

## 4. The three soak blockers

**Blockers 1 and 2 are closed on evidence:**

- **VM101 address stability** — router-side DHCP reservation `bc:24:11:19:4f:24` → `192.168.3.177`,
  **confirmed by reboot survival**, which Beta and Claude Code both identified as the decisive
  check. *(The interface still reads `dynamic`; that is correct for a reservation and not evidence
  against it.)*
- **Worker code parity** — `miner/` and `utils/` deployed from `git archive 18a2419` to all three
  CT100s: **17/17 exact on each**, module provenance re-verified **by the same method that found
  the drift** (fresh interpreter under `~/rocm_env`, `sys.modules` after importing
  `miner.range_miner_worker`). `git diff 18a2419 HEAD -- miner utils` is empty, so the deployed
  bytes equal HEAD's content.

**Blocker 3 — kernel-log observability — is NOT closed, and the soak launches without it by owner
order.**

**Alpha does not dispute Beta's reasoning.** The consequence is stated plainly and will be honoured
in the report:

> **The "no `GCVM_L2_PROTECTION_FAULT` / no GPU reset" criterion will be reported `UNAVAILABLE`,
> never `PASS`.** It was not checked. An inaccessible surface is not a clean one.

**Substitute detection, polled at 60 s and logged as a series:** `rocm-smi` device count and
per-GPU state on each rig · worker process liveness · repeated lease expiries per worker identity.
These detect *that* a GPU or worker died on a named rig. **They cannot classify the fault**, and no
classification will be claimed from inside the container — the Proxmox host logs persist and remain
readable from the console afterwards if it ever arises.

**The owner's technical basis, which Alpha shares:** `GCVM_L2_PROTECTION_FAULT` was a **PWC
launch-storm defect** — ~17K kernel launches per trial — which followed that workload across every
transport, a code revert and a package rollback. **RANGE-MINER's persistent per-GPU daemons
eliminate the workload**, and Phase 6.0 produced no reset and no fault on either CUDA or ROCm.

**Alpha records one qualification, once:** PWC was also stable at reduced scale and failed **only**
at full saturation. This soak is the first time RANGE-MINER meets that condition — which is what
Phase 7 exists to establish. The expectation is that it holds; the soak converts that into evidence.

## 5. Launch configuration

`n_parallel=1` **binding** · **exactly 25 frozen identities** (`bea580e76490`) · 50 trials · mixed
constant/hybrid · ≥5 high-survivor and ≥5 low-survivor · `serial_reference` (**§17 promotion is
Phase 6's, not this soak's**) · `resume_checkpoint` empty · clean tree at launch.

**The headline result is not throughput.** This is the first execution in the project's history with
`_FLUSH_CLEAR_IN_MEMORY = True`. **Whether the candidate list stays bounded across 50 trials is
what the soak is for**, and the report will carry the RAM series, not a peak.

Checkpoint census baseline for D6.3: **25 directories, 50 files, 266,835 bytes** — the delta will be
recorded.

## 6. VIR declaration

**Audit scope:** live hosts (VM101 + three CT100 workers) and the repo at `20ba2a6`.
**Searched surfaces:** `distributed_config.json`, `rig_profiles_config.json`,
`dataset_provisioning.json`; `nvidia-smi -L`; `ip addr` after reboot; per-file sha256 across all
four machines; `sys.modules` in a real interpreter on each rig. **Unavailable surfaces, and this is
the one that matters:** **the Proxmox host kernel logs on `.121`/`.155`/`.163` — no root key auth
from VM101.** That surface is why §4's criterion is `UNAVAILABLE` and not `PASS`.
