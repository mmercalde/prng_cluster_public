# CLAUDE CODE REPORT — S172 GATE-12 STALL (READ-ONLY DIAGNOSIS)

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`.
**Date:** 2026-08-07. **Author:** Claude Code (Team Alpha).
**Constraint honoured:** nothing was launched. No pipeline, no fleet, no worker, no bind on
5700, no `gate_s172_prod_shape.py`, no commit, no push, no production-code edit. Every
command was a read: SQLite `mode=ro`, `cat`/`ls`/`stat`, `git`-free source reads, and
read-only SSH probes (`date`, `ps`, `cat`, `ls`, `wc`).

---

## 0. HEADLINE — the brief's framing is wrong, and the correction matters

**This was not a dispatch stall. Work was dispatched, claimed, mined and completed.**

Alpha's leading hypothesis — quarantine on admission — is **REFUTED by the ledger**, both
branch (a) execution-set identity mismatch and branch (b) capability/variant mismatch.

> **Root cause: a staging-capacity DEADLOCK.** The run exhausted
> `staging_high_water_files` (default **512**, `miner/range_miner_coordinator.py:224`) partway
> through its second stage. Held staging reservations are released only by a Phase-5 ack path
> that **has no production caller**, or by `abort_trial` — which is terminal. Within a running
> trial, held reservations are a **one-way ratchet**. Trial 1 needed **1,028** staged shard
> files against a hard ceiling of **512**, so it could never finish. The four staging worker
> threads then spun on a 50 ms back-pressure retry loop that has **no timeout**, which is the
> 60.1 % CPU.

Two independent aggravating facts, both evidenced below:
- **The bound is not operator-settable.** `staging_high_water_files` has no CLI flag, no
  manifest key, and nothing anywhere assigns it. It is always 512.
- **The 600 s capacity timeout did not fire**, and could not: it keys off *paused connections*,
  a state this stall shape never enters. The process sat wedged for **~19 minutes** — well past
  the 600 s bound — and never reached a terminal.

---

## 1. THE LEDGER — the decisive artifact

**Two ledgers exist; the repo-root one is stale and would have misled.** `--staging-dir
/home/michael/miner_staging` in the EXEC CMD is authoritative for this run.

| ledger | last write | content |
|---|---|---|
| `~/distributed_prng_analysis/miner_ledger.db` | Aug 4 18:09 (WAL Aug 7) | trials from **2026-08-05** — NOT this run |
| **`/home/michael/miner_staging/miner_ledger.db`** | **Aug 7 18:04:18** | **the run diagnosed here** |

### 1.1 Workers — hypotheses (a) and (b) are refuted

```sql
sqlite3 "file:/home/michael/miner_staging/miner_ledger.db?mode=ro" \
  "select status, count(*) from workers group by status;"
eligible|25

  "select coalesce(quarantine_reason,'<null>'), count(*) from workers group by 1;"
<null>|25
```

**25 rows, all `eligible`, every `quarantine_reason` NULL. Zero quarantined.**
`zeus-ubuntu-vm:gpu0` (cuda) + 8 each on `rrig6600` / `rrig6600b` / `rrig6600c` (rocm),
registered 18:01:26 → 18:03:40.

- **(a) execution-set identity mismatch — REFUTED.** Every worker that connected was admitted
  eligible. No admission refusal was recorded for any identity.
- **(b) capability/variant mismatch — REFUTED.** `_validate_caps` rejected nothing (no
  quarantine reason exists), and `can_assign_variant` demonstrably returned True: stripes were
  assigned and completed for `java_lcg` **and** `java_lcg_reverse`.

### 1.2 Stripes — work was assigned and mined to completion

```sql
select run_id, state, phase, family_name, count(*) n,
       sum(substripes_done) sd, sum(expected_substripes) es from stripes group by 1,2,3,4;
distributed_config_t1_25e4f207|done   |1|java_lcg        |16|504|504
distributed_config_t1_25e4f207|staging|2|java_lcg_reverse|16|524|524
```

**Stage 0 (`java_lcg`): 16/16 stripes `done`, 504/504 sub-stripes.**
**Stage 1 (`java_lcg_reverse`): 16/16 stripes mined out — 524/524 sub-stripes, every stripe
`stripe_complete_seen=1` — but all 16 stuck in state `staging`, none reaching `done`.**

`staging` is the Blocker-5 state meaning *"the GPU is free but transfers may still run"*
(`range_miner_coordinator.py:1452-1457`). The GPUs finished. The transfer/staging side did not.

### 1.3 Shards and reservations — the deadlock, exactly

```sql
select stage, staging_status, phase5_status, count(*) from shards ... ;
st0|verified|enqueued|504
st1|pending |none    |516
st1|verified|none    |  8

select status, count(*), sum(size_bytes) from reservations group by status;
held|512|84498
```

- **512 reservations `held`. Zero released. Ever.**
- 504 (st0 verified) + 8 (st1 verified) = **512 exactly**.
- **`staging_high_water_files: int = 512`** — `miner/range_miner_coordinator.py:224`.

The match is exact, not approximate. The 513th reservation was refused and every subsequent
one with it, leaving **516 stage-1 shards permanently `pending`**.

Note the bytes: **84,498 total**, against a 16 GiB byte high-water. **This is not a
size problem. It is purely the file-count ceiling.**

### 1.4 Arithmetic — the trial could never have fit

From the EXEC CMD (`logs/gate12_prodshape_20260807_180116.log`, the run's last line):

```
--max-seeds 1073741824  --miner-stripe-size 67108864
--seed-cap-nvidia 5000000  --seed-cap-amd 2000000
```

| quantity | value |
|---|---|
| stripes per stage | 1,073,741,824 / 67,108,864 = **16** |
| sub-stripes, AMD stripe | ceil(67,108,864 / 2,000,000) = **34** |
| sub-stripes, NVIDIA stripe | ceil(67,108,864 / 5,000,000) = **14** |
| stage 0 shard files | 2×14 + 14×34 = **504** |
| stage 1 shard files | 1×14 + 15×34 = **524** |
| **trial total** | **1,028** |
| **ceiling** | **512** |

**Stage 1 alone (524) exceeds 512.** This is load-bearing for the fix: releasing capacity at a
*stage* boundary would not be enough. And this was **trial 1 of 8** (`--trials 8`).

---

## 2. WHY CAPACITY IS NEVER RELEASED — source trace

`reserve()` grants only if both marks hold, else returns None (back-pressure):

```python
if held_bytes + size_bytes > high_water_bytes: return None
if held_files + 1     > high_water_files: return None
```
— `range_miner_coordinator.py`, `LedgerStore.reserve`, and its docstring states the rule
plainly: *"A held reservation counts until it is explicitly released (ack + local delete, or a
failure-path cleanup) — **never on mere enqueue**."*

The release paths, and their production callers:

| release path | anchor | production callers |
|---|---|---|
| `ack_by_event_id` (L6, the real one) | `:4096` | **NONE** — tests only |
| `release_after_ack` | `:4060` | **NONE** — tests only |
| `ack_shard` | `:4052` | **NONE** — tests only; and it is explicitly *"Stubbed Phase-5 ack … Mere ack does NOT release capacity."* |
| `cleanup_reservation` / `cleanup_attempt` | `:4078`, `:4116` | failure paths only |
| `abort_trial` | `:4402-4404` | **terminal** |
| `commit_trial` | `:4293` | **does not release at all** |

Verified by grep across the tree excluding `tests/` and `MagicMock/`: `ack_by_event_id`,
`ack_shard` and `release_after_ack` appear **only at their own definitions**.

**This is by design, not an oversight, and that is what makes it a real architectural
constraint rather than a missing call.** The only thing production does is *enqueue*:

```python
if self.phase5_sink is not None:
    self.phase5_sink.publish_shard(manifest)
self.ledger.mark_shard_enqueued(...)     # :2406-2408
```

and `AssemblingPhase5Sink.publish_shard` (`miner/range_miner_npz_writer.py:1172`) states:
*"NO spool I/O happens here — publish stores a canonical DEEP COPY of the manifest and nothing
else; **every staged-spool read happens at commit-time assembly**."*

**Phase 5 needs the staged files to still exist at commit.** So they genuinely cannot be
released mid-trial under the current design — which means a trial's peak retained-file count
is inherently its *whole* shard count (1,028 here), and 512 was never survivable.

This matches the ledger exactly: 504 stage-0 shards sit at `phase5_status='enqueued'`,
`phase5_acked_at` NULL, `local_cleanup_status='none'` — enqueued, never acked, never freed.

---

## 3. WHERE THE 60 % CPU WENT

`_run_staging_job`, `miner/range_miner_coordinator.py:3933-3957`:

```python
while True:
    try:
        ... stage_inline_shard(...) ...
        break
    except StagingBackPressure:
        if not self._attempt_live_locked(run_id, stripe_id, attempt):
            self._release_admission(run_id, stripe_id, attempt); return None
        time.sleep(0.02)      # :3956
        continue
```

A **50 Hz unbounded retry loop**, executing `_attempt_live_locked` — a SQLite read — on every
pass. Its own comment is explicit that this is deliberate and untimed: *"an admitted attempt's
sub-stripe that cannot reserve yet WAITS and resumes … it is **NOT** timed out into the retry
matrix."* It exits only if the attempt stops being live. The trial never went terminal, so the
attempt stayed live, so the loop never exited.

With `--staging-workers 4`, up to four such threads spin concurrently; the per-connection pause
loop at `:3454-3459` adds another 50 ms poll per paused reader. That is a coherent account of
**60.1 % CPU in state `Ssl` with zero log output**.

*Confidence note:* the process is dead and was never profiled, so this is identified from
source plus the ledger state (we know for certain ≥1 job was in `StagingBackPressure`, because
the 513th reserve was refused). I did not measure it live and am not claiming I did.

---

## 4. WHY NOTHING EVER TERMINATED — a second, independent gap

`staging_capacity_timeout_expired()` (`:3509-3545`) measures the age of the **oldest currently
paused connection**:

```python
oldest = min((r["since"] for r in self._paused_connections.values()), default=None)
if oldest is None or (now - oldest) <= limit:
    return False
```

**If no connection is registered as paused, `oldest is None` and the timeout returns False
forever.** The wedge here lived in the *staging executor* — results already received, decoded
and recorded — not in a paused reader.

The timing proves the bound did not bind:

| event | time |
|---|---|
| last reservation granted (the 512th) | 18:04:06 |
| last shard row recorded | 18:04:18 |
| ledger WAL truncated (process death) | **18:23:14** |
| **stall duration** | **~19 min** |
| `--staging-capacity-timeout` | **600 s = 10 min** |

The trial row is still `state='running'`, `abort_cleanup_status='none'`,
`commit_delivery_status='none'`. **It sat ~19 minutes against a 600 s bound and never
terminated.** Michael's kill was the only terminal.

So the Beta-mandated bounded wait (§2.19) is real but **does not cover this stall shape**. Even
a correctly-sized file high-water would leave this hole open.

---

## 5. THE RIG-SIDE VIEW

Read-only SSH to `.122` / `.156` / `.164` (rigs run **UTC**, VM101 runs **PDT**, +7 — the rig
log stamps of `Aug 8 01:0x` are `Aug 7 18:0x` local, i.e. this run, not a later one):

| rig | `/tmp/minerlogs` total lines | gpu0.log content |
|---|---|---|
| rrig6600 (.122) | 15 | `Compiled kernel: java_lcg`, `java_lcg_reverse` |
| rrig6600b (.156) | 7 | `Compiled kernel: java_lcg_reverse` |
| **rrig6600c (.164)** | **0** | **empty** |

**No worker logged a registration ACK, a quarantine notice, or an assignment** — the worker log
only ever records kernel compilation, so its silence is *not* evidence of missing work. The
ledger is the only assignment record, and it shows work was assigned and completed.

**`rocm-smi` showing 0 % on .122 was workers sitting IDLE HAVING FINISHED**, not workers
starved. Ledger: `rrig6600:gpu0-6` completed 14 stage-0 stripes and `gpu0-7` completed 8
stage-1 stripes.

**Worth flagging separately: 9 of 25 workers never received any work at all.** With
`--max-seeds 1073741824 / --miner-stripe-size 67108864` there are only **16 stripes** for 25
daemons. All of `rrig6600c` (8 GPUs) plus `rrig6600b:gpu7` were idle by construction —
consistent with rrig6600c's zero log lines. **Even without the deadlock, this run would not
have exercised 25-GPU saturation.** Gate 12's stated shape should be reconciled with the seed
budget before the next attempt.

*Verified:* no miner workers are running on any rig now. (The first probe's `pgrep -c -f
range_miner_worker` returned 1 on each rig — that is the **documented self-match false
positive**; re-run as `ps ... | grep -e "[r]ange_miner_worker"` it returns nothing on all
three.)

---

## 6. THE BOUND IS NOT OPERATOR-SETTABLE — §2.7 defect class, eighth instance

```
window_optimizer_integration_final.py:1468
    staging_high_water_files = getattr(coordinator, 'staging_high_water_files', 512),
```

- **No CLI flag.** `window_optimizer.py` defines `--staging-dir`, `--staging-workers`,
  `--staging-queue-depth`, `--staging-deferred-max`, `--staging-capacity-timeout`. There is no
  `--staging-high-water-files` and no `--staging-high-water-bytes`.
- **Nothing assigns the attribute.** `window_optimizer.py:787-811` sets `staging_dir`,
  `staging_workers`, `staging_queue_depth`, `staging_deferred_max`,
  `staging_capacity_timeout` — and **not** either high-water.
- **No JSON sets it.** `/bin/grep -rn "high_water" --include=*.json .` → no hits (searched with
  `/bin/grep`, since the shell `grep` wrapper honours `.gitignore` and would skip `*.json`).

**The `getattr` default is therefore the only value this knob ever takes: 512.** This is
precisely the §2.7 silent-no-op shape — a control that exists, is read, and can never receive a
production value — and precisely the three-hop gap (§2.15) that the S172-BP remediation closed
for the *other four* staging controls while leaving these two behind.

It is also the §2.12b failure mode repeating: **a run admitted work that was unmeetable by
construction.** `_attempt_exceeds_highwater` (`:2875`) fail-fasts when *one attempt's* file
footprint exceeds the high-water (34 ≤ 512, so it passed), but **nothing checks the trial's
cumulative footprint (1,028) against the same ceiling.**

---

## 7. MINIMAL FIX — DESCRIBED ONLY, NOT IMPLEMENTED

No code was changed. Four separable items; **(1) and (2) are needed to run gate 12 at all.**

**(1) Give the file high-water an operator route, and size it to the trial.**
Wire `staging_high_water_files` (and `_bytes`) through the same manifest → CLI → coordinator
route the other four staging controls now use — `window_optimizer.py` argparse +
`coordinator.<attr> =` assignment — so the `getattr` default stops being the only reachable
value. For this gate-12 shape it must be **≥ 1,028**, with headroom.
*This alone is a bound-raise, not a correctness fix — it moves the wall.*

**(2) Fail closed BEFORE dispatch on a trial that cannot fit.**
Add a trial-scoped sibling to `_attempt_exceeds_highwater` (`:2875`), computed at stage
assignment from the frozen execution set + stripe geometry (stripes × per-worker sub-stripe
counts, summed over all stages). A run that cannot fit should terminate with a named
`coordinator_staging_...` reason **before the first stripe is dispatched** — never deadlock 19
minutes into a trial. This is the §2.12b lesson applied to staging capacity, and per the
**owner rule (§7)** it is the structurally stronger of the two — the property holds by
construction rather than by an operator having picked a large enough number.

**(3) Close the capacity-timeout gap.**
`staging_capacity_timeout_expired()` (`:3509`) must also observe staging jobs wedged in
`StagingBackPressure`, not only `_paused_connections`. As written, the bounded wait Beta
mandated in §2.19 cannot fire for this stall shape at all.

**(4) The architectural question — BETA DECISION, not Alpha's to take.**
Capacity is released only at `abort_trial`; `commit_trial` releases nothing; the L6 ack path
has no production caller **because `AssemblingPhase5Sink` reads staged spools at commit-time
assembly and needs the files until then**. So a trial's retained-file count is inherently its
total shard count. Either Phase 5 assembles incrementally and acks as it consumes (freeing
capacity mid-trial, which is what `ack_by_event_id` was built for), or the high-water must be
sized to the whole trial by construction. **This changes the Phase-5 boundary contract and
belongs to Beta**, and it interacts with the D5 §6.7.A retention rules — I am not proposing a
mechanism here.

---

## 8. IS A FURTHER RUN NEEDED?

**No run is required to establish the root cause.** The ledger is dispositive and was read with
everything stopped, exactly as the brief anticipated.

A run **is** required to validate any fix, and it is Michael's to authorise and launch. When
that happens, the minimum useful shape:

- the **same** gate-12 command, with the file high-water raised above the per-trial shard count
  (≥1,028 for this seed budget) — this alone tests whether the deadlock is the *only* wall
  between here and a completed trial;
- watch for `[S172-BP] summary` reporting `staging_jobs_completed` climbing past **512**, which
  is the single number that proves the ceiling is cleared;
- reconcile the seed budget against the intended stripe count first, or 9 of 25 daemons will
  again sit idle and the run will not test saturation.

---

## 9. WHAT I COULD NOT DETERMINE

Stated plainly, per the brief — an unproven claim is worse than a gap.

1. **How the 516 unstaged stage-1 shards were distributed between the in-memory `_deferred`
   queue and reader-paused connections.** Both are process memory and died with the kill;
   neither is in the ledger. I can prove all 524 shard *rows* were recorded (so all 524 results
   were decoded and received), and that only 8 ever reserved — but not the internal split. This
   does not affect the root cause: under either distribution the work waits on capacity that
   can never be released.
2. **Why the `coordinator_staging_capacity_invariant` path (`:3835`) did not fire.** With a
   derived deferred bound in the low hundreds and 516 shards unable to stage, I would have
   expected `_defer_locked` to refuse and terminate the trial. It did not — the trial is still
   `running`. The most likely explanation is that the §1 reader pause held frames before the
   deferred queue overflowed, but **I cannot evidence that from a dead process** and am not
   asserting it.
3. **Whether the 60.1 % CPU was exclusively `_run_staging_job`'s loop.** Identified from source
   and ledger state, not from a live profile (§3).
4. **The `GCVM_L2_PROTECTION_FAULT` / GPU-reset criterion: `UNAVAILABLE`, not PASS.** CT100 is
   an unprivileged LXC and VM101 has no root key auth to the Proxmox hosts (§2.17). Not
   checked, and an inaccessible surface is not a clean one (VIR-1/VIR-5). Nothing in this
   diagnosis depends on it — the GPUs completed all assigned work.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every claim carries either a `mode=ro` SQLite result or a `file:line`
  read live this session; timestamps cross-checked against ledger, WAL mtime and log size
  (6,211 bytes, matching the brief's frozen figure exactly).
- **clean control:** the stale repo-root ledger was read first and correctly identified as a
  different run (2026-08-05) before being set aside — it would have shown 25 eligible workers
  and 0 quarantined too, for a different run, which is exactly how a stale artifact misleads.
- **fault-injection control:** none run — this is read-only diagnosis; no detector was exercised.
- **completion sentinel:** all five brief items answered; gaps enumerated in §9.
- **unavailable-observer behaviour:** rig GPU kernel logs report **UNAVAILABLE** (§9.4), never PASS.
- **audit claim scope:** one run — `distributed_config_t1_25e4f207`, 2026-08-07 18:01:20 PDT.
  No claim about earlier attempts beyond what the brief states.
- **searched surfaces:** `/home/michael/miner_staging/miner_ledger.db` (ro) ·
  `~/distributed_prng_analysis/miner_ledger.db` (ro) · `miner/range_miner_coordinator.py` ·
  `miner/range_miner_npz_writer.py` · `miner/step1_ingress.py` · `window_optimizer.py` ·
  `window_optimizer_integration_final.py` · `logs/gate12_prodshape_20260807_*.log` (4 files) ·
  `agent_manifests/*.json` and all repo `*.json` via `/bin/grep` · live rig filesystems on
  .122/.156/.164 · live process table on VM101 and all three rigs · `docs/` and the governance
  trail via the `tfm-project-facts` skill (§2.7, §2.12b, §2.15, §2.19, §4).
- **unavailable surfaces:** the dead `window_optimizer.py` process (pid 10460) — no profile, no
  core, no captured stdout; coordinator in-memory state (`_deferred`, `_paused_connections`,
  `_bp` counters, `[S172-BP] summary`) — never flushed to any log; Proxmox-host kernel logs on
  .121/.155/.163 — no root key auth from VM101.
- **governance trail searched:** `tfm-project-facts` v16 §2.7 (silent-no-op class), §2.12/2.12b
  (admission liveness, unmeetable-by-construction), §2.15 (three-hop parameter route), §2.19
  (S172-BP law and mechanics), §4 (frozen surfaces), §7 (owner rule on structurally stronger
  mechanisms). Commit `27ae7a9` message read for the S172-BP amendment scope.
- **chapters searched:** none — no claim here concerns sieve mathematics or pipeline semantics.
