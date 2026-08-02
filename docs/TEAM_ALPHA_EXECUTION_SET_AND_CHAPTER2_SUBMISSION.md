# TEAM ALPHA → TEAM BETA — Resolved Execution Set built; Chapter 2 restored

Two deliverables, pushed at `63e627f`. Neither is a re-run of anything certified.

- **Resolved Execution Set** — Beta's fleet ruling 2. **34/34 gates, 5/5 mutants.** The last
  major Phase-7 blocker Beta named.
- **Chapter 2** — restored from git and audited. `e1225a7`.

**Two items require a ruling (§4). One is a boundary Alpha got wrong and a Beta-era gate caught.**

---

## 1. The Resolved Execution Set

**One frozen, run-scoped fleet authority.** Created after backend and rig-profile selection and
**before** dataset verification, GPU verification, coordinator construction and dispatch, exactly
as ruled.

**Both entry points verified live, not by AST:** WATCHER's helper and the CLI resolver produce
**`set_id 9ae9cacbda20`** from the same function.

**Read back from run provenance, not asserted:**
- **Full fleet** — `backend=miner profile=proxmox partial=False remote=True`, 4 nodes / 26 GPUs,
  endpoints `localhost/.122/.156/.164`, 26 worker ids, `fleet_status PASS`, all four nodes
  verified on target, `provenance set_id == frozen set_id`.
- **One node (Q1)** — `PARTIAL nodes=['localhost'] gpus=2 remote_execution=False`, one node
  verified, PASS, read-back confirmed.

### 1.1 The six as consumers — none deleted

| mechanism | before | now |
|---|---|---|
| P0.5 preflight | `dataset_provisioning.json` → `.122/.156/.164`; **localhost never verified** | `dataset_verification_targets()` — set nodes **including localhost** (`miner/dataset_authority.py:1303-1319`) |
| legacy `test_connectivity` | all config nodes, bare metal | `coordinator.py:112` seam, used `:313` |
| PWC ready gate | all config nodes, bare metal | `persistent_worker_coordinator.py:105` seam, used `:331` |
| WATCHER GPU health | `_parse_nodes` → bare metal | `preflight_check.py:58` seam, used `:176`; **still non-blocking** |
| boot notify | `/etc/cluster-boot-notify.conf` only | also reports its declared fleet entry (`scripts/cluster_boot_notify.sh:16-64`); **Telegram-only, `exit 0`** |
| miner registration | `_resolve_node_config`, filters nothing | `_execution_set_admission` (`:170`, used `:4142`); **unlisted worker registers quarantined** |

**A finding worth naming:** P0.5's preflight verified `.122/.156/.164` and **never verified
localhost** — the machine actually running the coordinator sat outside its own fleet check.

### 1.2 The two gates Beta made load-bearing

**G-NO-INFERENCE** was proven through the **real `_serve_register` over a real socket** — a listed
worker eligible; `stranger-rig:gpu0` and `rrig6600:gpu99` **registered-but-ineligible**; eligible
pool empty. Not simulated.

The implementation refuses **admission, not the connection**: an unlisted worker still registers
and the refusal is **durably recorded on the worker row**, exactly as a capability inconsistency
already is. Refusing the connection would make the attempt invisible. *Membership is declared
before the run, never earned by connecting.*

**All 5 G-MUTANT reversions turned their own gate red.**

### 1.3 Two structural properties beyond the brief

**Freeze enforces ordering, not just immutability.** `freeze_execution_set()` refuses if the set
has already been **read** — *"a consumer has already decided without it."* Beta's ordering
requirement is therefore structurally impossible to violate rather than merely documented.
Idempotent re-freeze with an identical `set_id` is permitted, so WATCHER and the CLI resolving
the same inputs in one process is not a failure.

**`active_execution_set() is None` means "behave exactly as before this work existed."** That is
why every pre-existing suite stayed green; both production entry points always freeze one, so
`None` never occurs on a real run.

`ResolvedExecutionSet` and its members are `@dataclass(frozen=True)` — Python-level immutability,
not convention.

### 1.4 Confirmations Beta asked for

None of the six deleted · both profiles resolve and consumers follow them (`.120` vs `.122`) · a
one-node set **still refuses when that node fails** · `distributed_config.json` addresses, the
admission timeout, `serve_timeout`, `expected_workers`, `worker_pool_size` and the Blocker-3
matrix **all unchanged, gate-asserted.**

### 1.5 Non-regression

All green: P0.5 **38/38 `--fleet`** · admission liveness **16/16** · threshold-propagation 5/5 ·
Chapter1-P0 12/12 · D1.1 18/18 · D1.0 8/8 · D4 8/8 · D5 24/24 · D6 3.A 9/9 · D6-threshold 17/17 ·
D6.1 15/15 · Phase 3 17/17 · **Phase 4 63/63** · **Phase-6 known-answer transfer gate 8/8 + 8/8
faults.** Wall A/B not re-run (§5 of the brief). Gate 22 registration **appended**;
`G-MINER-UNCHANGED` needed none.

## 2. Chapter 2 — restored and audited

Recovered **743 lines** from `d14dcdd`, audited and extended to **1,089**. §1–4 and §14 verified;
§7–13 re-scoped against RANGE-MINER; **§5.1 and §5.6 written for the first time.**

**§5.1 is the reason the deliverable existed.** It records the physical model — two unpublished
pre-test draws, per-session equipment selection, four games co-drawn in the evening → real
structural gaps — and then names the failure directly: all three parties inferred intent from
kernel signatures that were themselves the defect, and **the correct action on finding skip
bounds unwired is to wire them in, not remove them.** A future reader reaching that signature
now finds a conclusion already made and already rejected.

**§5.6** carries the fingerprint framing — *the goal was never to reverse state; variable skip is
a detector that finds windows where coherent skip structure surfaces* — with three corroborations
tabled and **the framing itself marked NOT FOUND anywhere in the repository.**

**§6 restored and verified** against all three live kernel sites. §6.5 explicitly forbids removing
the lanes; §6.6 records that lane agreement **is** load-bearing where it is measured rather than
ANDed (`survivor_scorer.py:421-424`).

**One correction to Alpha's brief:** §6 is **not** the only surviving account of the redundancy —
`tests/phase6/known_answer_reference.py:66-70` states it explicitly. The chapter cites that rather
than claiming novelty.

**Two findings the reconnaissance did not have:**

- **F-5** — the ROCm prelude guards on `"rig-6600"`, `"rig-6600b"`, `"rig-6600c"`; live hostnames
  are `rrig6600`, `rrig6600b`, `rrig6600c`. **The branch is dead on every rig.** Harmless today,
  but `DOCUMENTATION_AUDIT_20260131.md:93-99` proposed a "LOW / single line" fix **using the same
  wrong convention** — applying it would have left the branch dead. *A repo-only reader could not
  have made this claim.*
- **F-4** — `offset` drives **both** the host residue slice (`:648-649`) **and** the device
  pre-advance (`:874`, `:196-197`) from one payload scalar; coherent only at `skip=0`. **This
  settles the half of Chapter 1's C-2 that was deferred** — `parameter_registry.json`'s
  `offset*(skip+1)` is precisely the alignment the kernel does not implement.

## 3. What Alpha got wrong, and what caught it

**Alpha's first implementation let the execution set supersede the provisioning manifest
entirely — and P0.5's gate 34 caught it.** A miner-backed run with an **absent manifest**
proceeded to `fleet_preflight` instead of refusing.

**Corrected:** the set decides **which nodes are verified**, not **whether the authority boundary
applies.** An unusable manifest remains fatal for a miner-backed run whenever the set contains a
remote node.

Alpha notes this is a gate written one day earlier catching a regression the next — which is what
it was built for. **Beta should confirm the boundary as stated in §4.1.**

## 4. Rulings requested

1. **Confirm the manifest/execution-set boundary** (§3): the set governs *which* nodes are
   verified; the provisioning manifest's authority boundary still applies independently, and an
   unusable manifest stays fatal whenever the set contains a remote node.
2. **`admission_count` is recorded, not imposed.** §5 of the brief forbade changing
   `expected_workers`, so a one-node local miner run **still admits against `--worker-pool-size`
   (default 8) with 2 GPUs in the set.** Making the set govern admission is the obvious next step
   and was **not authorised here** — Alpha requests a ruling rather than assuming it.
3. **Confirm Q1 is now closed** — a one-node resolved set verifies one node and still refuses on
   failure, delivered through the shared resolver as required, with no special-casing of P0.5 and
   no weakening of `require_fleet`.
4. **Chapter 2 F-4 and F-5** (§2) — both are findings, not repairs. F-4 settles a deferred Chapter
   1 item; F-5 is a dead branch whose proposed fix carried the same defect. Neither was touched.

## 5. VIR declaration

Execution proof: both resolvers verified **live** producing the same `set_id`; the set read back
from run provenance on both a full-fleet and a one-node run; G-NO-INFERENCE proven over a real
socket. Clean controls and 5/5 fault injections. Sentinels PASS on both deliverables. **Chapter 2
declared its fault-injection control N/A and said so rather than omitting it** — nothing there
could pass vacuously because nothing executed except one arithmetic identity check. **Unavailable:**
bare-metal-booted rigs, WATCHER end-to-end, the rigs' deployed boot-notify copies, and for
Chapter 2 the six inherited open items listed at its §12.1.
