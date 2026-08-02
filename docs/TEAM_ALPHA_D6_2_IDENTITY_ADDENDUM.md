# TEAM ALPHA → TEAM BETA — addendum to D6.2 REV2, while it is under review

**Re:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md` REV2 (`285cbd7`).

**Why this is arriving mid-review rather than after.** D6.3 turns out to depend on a field D6.2 is
already positioned to add, in the exact section Beta is reading. Raising it now costs one
paragraph of Beta's attention; raising it after approval costs an amendment cycle on the Phase-7
critical path.

**Nothing in REV2 is withdrawn.** This adds one field to §3 and one gate to §6.

---

## 1. The finding

`.s172_checkpoint/<run_id>/` and the published generation use **two different run identities**,
both defined in `window_optimizer_integration_final.py`, roughly 2,150 lines apart:

| | identity | anchor |
|---|---|---|
| checkpoint directory | `_flush_run_id()` → `f"{hostname}-{pid}-{int(time.time())}"` | `:448`, `:452-455` |
| published generation | `run_id=f"step1_{prng_base}_{int(seed_start)}"`, marked `[S142-C] canonical` | `:2606` → `generation_id`, `utils/run_finalizer.py:1652` |

`PRNG_CHECKPOINT_RUN_ID` (`:381`) can override the first, and **the only setters found are test
harnesses** — `test_s172_phase5_d3_25_candidate_ingress.py:217`,
`test_s172_phase5_d6_production_adapter.py:952`, `test_s172_d6_1_flush_durability.py:238`. No
tracked production code sets it.

**Consequence: nothing on disk correlates a checkpoint directory with the generation its run
published.**

## 2. Why that blocks D6.3 specifically

Beta's D6.3 constraint: *never remove active, unresolved or audit-retained state merely for
exceeding an age or count threshold.*

That constraint requires a **resolution signal** — some way to establish that a given checkpoint's
run reached a certified generation and the checkpoint is therefore superseded rather than
abandoned mid-flight. **Today that signal does not exist.** A retention implementation could
observe only directory name and mtime, which are precisely the age/count inputs Beta ruled out.

So D6.3 is not "small but unwritten." **As specified, it is currently unimplementable**, and would
have been discovered as such by whoever picked it up — most likely partway through implementation.

## 3. The requested amendment to D6.2 — one field, one gate

D6.2 §3 already rewrites the transaction identity block (new schema version, widened digest, new
`encoding_version`). Alpha requests one further key:

- **`canonical_run_id`** — the `[S142-C]` value `step1_{prng_base}_{seed_start}`, recorded in
  every checkpoint member's identity block.

And, if Beta agrees the checkpoint should record its own outcome:

- **`published_generation_id`** — written back after `finalize_run` succeeds, absent while a run
  is in flight. **Absence then means unresolved**, which is exactly the state Beta's rule protects.

Plus **G-RUN-IDENTITY**: the recorded `canonical_run_id` matches the value the same run passes to
`finalize_run`, proven on a real run rather than by construction in the test.

## 4. What Alpha is NOT proposing, and why it matters

**Do not set `PRNG_CHECKPOINT_RUN_ID` to the canonical run_id.** It is the obvious fix and it is
wrong: `step1_{prng_base}_{seed_start}` is **not unique across reruns** — two runs over the same
family and seed range produce the identical string. Using it as the directory name would collide
their checkpoints and **break Beta's run-isolation condition 3**, which exists because consecutive
or concurrent runs must not be able to overwrite each other's snapshots.

**The directory stays process-unique. The identity goes in the block.** The two identities answer
different questions — *"which process wrote this?"* and *"which run does it belong to?"* — and the
defect is that only the first was ever recorded.

## 5. Rulings requested

1. **Add `canonical_run_id` to D6.2's identity block** (§3), with G-RUN-IDENTITY.
2. **`published_generation_id` — in or out?** In makes absence a positive unresolved signal and
   makes D6.3 straightforward. Out keeps D6.2's scope tighter and leaves D6.3 to derive
   resolution some other way, which Alpha has not identified.
3. **Confirm D6.3 sequences after D6.2**, on the dependency above, rather than in parallel.

## 6. VIR declaration

- **audit claim scope:** repo-scoped, `285cbd7`.
- **searched surfaces:** tracked repo — `*.py`, `*.sh`, `*.json`, `*.md`, `*.service`, `*.env`.
- **unavailable surfaces, and this one matters:** **host state on VM101 and the rigs.** A systemd
  unit, shell wrapper, WATCHER-injected environment or uncommitted local file could set
  `PRNG_CHECKPOINT_RUN_ID` in production, and no repo-scoped search can see it — the same class as
  the enabled `daily3scraper.service` whose absence Alpha once reported from a clone. **The claim
  "no production setter exists" is therefore [UNVERIFIED] pending host confirmation**, which the
  D6.3 investigation brief makes its first question. If a host setter does exist, §1's finding
  narrows but §4's collision hazard stands regardless.
