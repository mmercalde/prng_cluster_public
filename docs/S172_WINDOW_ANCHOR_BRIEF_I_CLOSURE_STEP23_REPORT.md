# BRIEF I — POST-COMMIT CLOSURE STEPS 2–3, REPORTED AGAINST `48a8705`

**Commit under proof:** `48a87059f5200e00727556f05c1462df07ba4614`
**Responds to:** `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md`, closure obligations 2 and 3
**Initiated by:** Michael, explicitly, 2026-08-22 (`CLAUDE.md` rule 3)
**Executed by:** Team Alpha on VM 101 (`zeus-ubuntu-vm`, `192.168.3.177`)

## VERDICT

| obligation | outcome |
|---|---|
| **2a — fleet source parity at the Brief-I commit** | ✅ **PASS — 30/30 MATCH · 0 MISMATCH · 0 UNAVAILABLE** |
| **2b — host/worker SCHEMA parity** (Beta: *"kernel hashes identical does not replace"* it) | ✅ **PASS — 3/3 rigs carry the Brief-I contract live** |
| **2c — production-shape proof (`G-PROD-SHAPE`)** | ⛔ **UNAVAILABLE** — required, attempted, could not be obtained |
| **3 — report against the commit hash** | this document |

**`UNAVAILABLE`, in Beta's own vocabulary, is fatal-for-the-topology, not `NOT_APPLICABLE`.**
The check was needed and could not be completed. It is **not** reported as a pass, and it is
**not** waived. **Brief I is therefore NOT presented as ready for final acceptance.**

---

## 1. Why the production-shape proof could not be obtained

`tests/gate_s172_prod_shape.py` was read at source, not from summary. **It is a VERIFIER, not a
driver** — `--log` is required, and it reads the log, staging ledger and published generation that
a *completed* production-shape run leaves behind. It cannot launch anything.

**No production run exists at `48a8705`.** The newest committed trial in the live ledger
(`/home/michael/miner_staging/miner_ledger.db`) is `distributed_config_t1_554463d3` — Gate-12
attempt 9, 2026-08-17, launched from **`e9ca800`**. Producing a run at `48a8705` is a pipeline
launch: Michael-initiated only, and explicitly out of scope for this session.

The gate was executed against the newest available run to obtain execution proof (VIR-1) and to
establish what it measures:

```
G-PROD-SHAPE vs attempt 9 (e9ca800)   18 pass / 4 fail / 0 unavailable of 22 legs
                                       COMPLETION SENTINEL: FAIL     EXIT=1
```

**That outcome carries NO authority for `48a8705`** — it describes a pre-separation run from a
different commit, and is recorded only as the baseline that produced §4's two findings.

## 2. Obligation 2a — fleet source parity: PASS

**Pre-deploy state, measured on target before anything was changed.** All three rigs were
uniformly at **`e9ca800`** — not mixed vintage this time:

```
rig (all three)   miner/range_miner_worker.py       043522e96b44855f   == e9ca800
                  miner/range_miner_coordinator.py  1fd8284e1219e009   == e9ca800
                  the other 8 governed files        already canonical
```

**A finding in its own right: the fleet had never received the field-6 repair `d8b21e3`** either
— that commit moved the coordinator to `53b5ce87c02f46c9` and the rigs still carried the
`e9ca800` bytes. A launch from any commit after 2026-08-20 would have been refused by the parity
wall. **That is the wall working as designed** (§2.44: a principle with no enforcing gate is not a
control).

**Deployment:** targeted `scp` from VM 101 of all ten governed files to all three CT100s —
all ten, not only the two that differed, so no assumption about rig state is load-bearing.
`mkdir -p` preceded the `scp` that fills it. Verified on target afterwards, each host printing its
own `socket.gethostname()`.

**The gate:**

```
governed files    10 (pinned; Beta minimum 5 included)
closure derived   miner/range_miner_worker.py -> 10 project files (AST, repo-local only)
local HEAD        48a87059f5200e00727556f05c1462df07ba4614  [CONTEXT ONLY — never an input]
local tree        governed files CLEAN
rows              30 MATCH · 0 MISMATCH · 0 UNAVAILABLE
GATE-12 PARITY GATE : PASS                                            EXIT=0
```

**The AST closure cross-check is a second result and is worth stating separately:** the worker's
statically reachable project-local closure derives to exactly the 10 pinned files. **Brief I
introduced no uncovered project import into the worker's closure** — had it done so, the gate
refuses and names the file.

## 3. Obligation 2b — host/worker SCHEMA parity: PASS

Beta: *"Kernel hashes being identical is necessary but does not replace host/worker schema
parity."* Digest parity proves the **bytes** arrived. This proves the **contract** is live in the
deployed module. `logs/brief_i_evidence/schema_probe.py`, executed under each rig's own
`~/rocm_env`, output written via `os.write(1, ...)` because importing `sieve_gpu_worker` replaces
`sys.stdout`.

| property asserted on the deployed module | rrig6600 | rrig6600b | rrig6600c |
|---|---|---|---|
| `miner.range_miner_worker` imports | ok | ok | ok |
| capability/policy symbols, all 10 present | ✅ | ✅ | ✅ |
| `GENERATOR_PHASE_V1_PIN == 0` | ✅ | ✅ | ✅ |
| `PHASE_CAPABLE` / `INCAPABLE` / arity = 20 / 4 / 24 | ✅ | ✅ | ✅ |
| `BuildContext` **has** `generator_phase` | ✅ | ✅ | ✅ |
| `BuildContext` **has no** `offset` | ✅ | ✅ | ✅ |
| legacy `{"offset": …}` **actively rejected** | `ResidueResolutionError` | idem | idem |
| missing `generator_phase` **refused, not defaulted** | `ResidueResolutionError` | idem | idem |
| python | 3.10.4 | 3.10.4 | **3.10.12** |

**0 failed checks on 3/3 rigs. Three distinct hostnames** — the §2.17 lesson that three machines
must not be one machine answering three times. The last two rows are behavioural, not
introspective: the probe *calls* the guards and records the exception class, so a module that
merely defines the symbols cannot pass.

**Carried forward, not new:** `rrig6600c` runs Python **3.10.12** against **3.10.4** on the other
two — open item 1 in the skill's §2.51, first recorded at the MP-1 deploy, **still undiagnosed**.
It does not gate the byte-parity contract, which governs deployed `.py` bytes, and all 30 digests
matched across both interpreter versions. Recorded because it was observed, not because it moved.

## 4. Two HOST-STATE findings — PRE-EXISTING AT `e9ca800`, not Brief I's

Both were surfaced by the G-PROD-SHAPE baseline run. **Both are exhibited by attempt 9, which
launched from `e9ca800`, nine commits and one entire Brief before `48a8705`.** They are reported
here so Beta rules on Brief I's code with the host-state issues correctly separated from it.

### 4.1 `G-PROD-SHAPE` is NOT RUN-SCOPED — three of its four failures are other runs'

Sections B and C aggregate the **shared cumulative ledger** and the **shared staging directory**
across every run in history. Scoped to attempt 9's own rows:

| leg | gate verdict | attempt-9-scoped | cause of the difference |
|---|---|---|---|
| B5 phase-5 acked | FAIL 11692/29082 | **5693/5693 PASS** | the `enqueued`/`none` rows belong to seven *aborted* runs |
| B6 local cleanup | FAIL 11692/29082 | **5693/5693 PASS** | same |
| B7 remote-delete | FAIL | **0/5693 — still FAIL** | genuine, see §4.2 |
| C2 staged-file leak | FAIL, 512 files | not attempt 9's | see §4.3 |

**Consequence: on this host `G-PROD-SHAPE` cannot reach PASS for any run, however clean, while
that residue and the B7 condition stand — and it gets harder over time**, because every future
aborted run adds rows the next run's verdict inherits. This is a property of the gate's scoping,
not of any run. **Recorded, not repaired.**

### 4.2 B7 — `remote_delete_status`: a writer EXISTS and has never fired

Read-only source check, per instruction to establish the category and stop.
Full evidence: `logs/brief_i_evidence/B7_remote_delete_status_source_check.md`.

```
:1265  DDL     remote_delete_status TEXT NOT NULL DEFAULT 'none'
:1666  WRITER  UPDATE shards SET remote_delete_status=?   in def set_remote_delete (:1657-1672)
:8270  reader  in retry_remote_delete

set_remote_delete <- 4 production call sites (AST, not text)
_delete_remote    <- exactly ONE caller: _finalize_stage at :4528
```

`_delete_remote` writes the column on **both** branches of its own try/except — `"deleted"` on
success, `"failed"` on any exception. **Had it executed once for any shard, that shard could not
still read `none`.** Measured: **0 of 29,082 shards across all nine runs**; **0 of 5,693** for
attempt 9 alone. Its one call-site guard is `if task.kind == "remote":` — so no `StagingTask` in
nine runs was classified `remote`, on a fleet where 24 of 25 workers are remote.

**Category: NOT a gate/observability defect.** The column is not unwritten by design; a live
production writer exists and has never run.

**Deliberately not pursued** (establish which, do not diagnose further): why `task.kind` is never
`"remote"`; whether remote spool files are consequently accumulating on the three rigs; whether
B7's assertion is correctly specified for this topology. **The second of those has an operational
consequence and is flagged for an owner decision, not acted on.**

**Attribution:** Brief I's coordinator diff (`e9ca800..48a8705`) contains **zero** occurrences of
`_delete_remote`, `_finalize_stage`, `set_remote_delete`, `remote_delete_status` or
`retry_remote_delete`.

### 4.3 C2 — 512 orphan staged payloads from a run with NO ledger trace

**Inventoried, nothing removed.** `logs/brief_i_evidence/C2_orphan_staged_payloads.json` carries
sha256, size and mtime per file.

```
files        512                     total 84,498 bytes
run id       distributed_config_t1_25e4f207   (all 512, one run)
mtime span   2026-08-08T01:01:46Z .. 01:04:06Z     (2 min 20 s)
ledger rows  0 in trials · 0 in stripes · 0 in shards · 0 in reservations
```

That window is **2026-08-08 — the day before Gate-12 attempt 1**, whose `689f3cd9` is the oldest
trial the ledger holds. The run predates the entire attempt sequence and left no trace in any of
the four tables.

**Disposition is Michael's and Beta's.** Removing the files would clear C2 and simultaneously
destroy the only surviving record of how a run escapes the ledger entirely.

## 5. What remains owed before Brief I can be ACCEPTED

Closure obligation 1 is complete (reported separately: phase4 63/63, cleantree 31/31, Brief I
25/25, R-1 44/44, MP-1 38/38, zero allowlist changes). Obligations 2a and 2b are **PASS**.

**Outstanding: a production-shape run at `48a8705`, followed by `G-PROD-SHAPE` against it.** That
is a pipeline launch and is Michael's to initiate. §4.1 and §4.2 mean the gate will not reach PASS
on this host as it currently stands even for a clean run — so a disposition on the orphan residue
and on B7 is a precondition for that proof, not an optional tidy-up.

**The Phase-7 soak cannot substitute** (Beta, and §2.54): it is non-certifying, and additionally
**pre-separation** — it cannot certify window-anchor semantics.

## 6. Verification-integrity controls (VIR-1…6)

- **execution proof:** every command wrapped `; echo "EXIT=$? SENTINEL_DONE"`; parity gate EXIT=0, G-PROD-SHAPE EXIT=1, both sentinels present in the saved logs.
- **clean control:** pre-deploy digests captured on target *before* any change; 2 of 10 files differed, proving the deploy was a real state change and not a no-op.
- **fault-injection control:** not applicable to a measurement pass; the parity gate's own refusal behaviour was not exercised.
- **completion sentinel:** parity `PASS`; prod-shape `FAIL` (attempt-9 scope); schema probe 3/3 rows returned with 0 failed checks.
- **unavailable-observer behavior:** `G-PROD-SHAPE` for `48a8705` reported **UNAVAILABLE**, never `NOT_APPLICABLE`, never a pass. Two CTs answered before the third; no parity claim was made until all three were up.
- **audit claim scope:** the ten governed files and the worker's Brief-I wire contract, on three CT100 workers, at `48a8705`. **No claim about GPU execution, kernel results, or any run.**
- **searched surfaces:** live VM 101 tree at `48a8705` · live SSH to all three CT100s · the live miner ledger · the staging filesystem · `git diff e9ca800..48a8705` · AST of the coordinator and worker.
- **unavailable surfaces:** any production run at `48a8705` (none exists); rig GPU kernel logs (unprivileged LXC, §2.17 — reports UNAVAILABLE, never PASS).
- **governance trail searched:** `TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` · `…_SCOPE_RULING.md` · `PROPOSAL_…_v1_1.md` · skill §2.53/§2.54/§6.
- **chapters searched:** Chapter 2 §7.2 (F-4 disposition), not modified by this pass.

## 7. Evidence

```
logs/brief_i_evidence/parity_gate_48a8705.log        the 30/30 wall, full 64-hex per row
logs/brief_i_evidence/parity_gate_48a8705.json       the gate's own evidence bundle
logs/brief_i_evidence/schema_probe.py                the probe, as executed
logs/brief_i_evidence/schema_probe_48a8705.jsonl     3 rows, one per rig
logs/brief_i_evidence/prodshape_attempt9_e9ca800.log the baseline run — NOT evidence for 48a8705
logs/brief_i_evidence/B7_remote_delete_status_source_check.md
logs/brief_i_evidence/C2_orphan_staged_payloads.json 512 files, sha256/size/mtime
```

`logs/` is ignored as a whole directory (`.gitignore:85`); the tracked tree stayed clean
throughout, so the clean-tree admission predicate and `W-NO-WEAKENING` are unaffected.

**Nothing was committed. Nothing was pushed. No pipeline was launched.**
