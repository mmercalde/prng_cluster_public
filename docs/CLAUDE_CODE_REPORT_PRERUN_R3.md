# TEAM ALPHA — PRE-RERUN R3 IMPLEMENTATION REPORT (P2 · S3 · S4 · E1)

**Host:** VM101 · repo `~/distributed_prng_analysis` · **HEAD `49ff9b4` throughout, not reverted.**
Every test run under `source ~/venvs/torch/bin/activate`.

**Nothing launched. No commit, no push, no fleet, no port-5700 bind, no real-rig SSH.**
Port 5700 confirmed UNBOUND at start and at end. Ledger mtimes unchanged at start and end:
`miner_ledger.db 2026-08-04 18:09:56.049862973`, `prng_analysis.db 2026-08-09 12:37:05.204068761`.

---

## 1. THE OPERATIVE BETA CHECKLIST — verbatim, with closure evidence per row

```
P1  CLOSED
P2  OPEN

S2  CLOSED
S3  PARTIAL — turnover logic good; combined authority line absent
S4  OPEN

E1  OPEN
```

| Beta row | Alpha closure | Evidence |
|---|---|---|
| **P1 CLOSED** | untouched, re-verified | `tests/test_preflight_gpu_probe.py` **12/12**; `preflight_check.py` byte-unchanged `sha256 cfbde94c…` (identical to R1 and R2) |
| **P2 OPEN** | **CLOSED** | Gate-12-only fail-close gate wired into `gate12_launch.sh` ahead of clean slate, sampler, coordinator and fleet; `tests/test_gate12_gpu_gate.py` **9/9**, full input space enumerated, fixture-driven, no rig contacted. §2 |
| **S2 CLOSED** | untouched, re-verified | the R2 D/E-series arms all green inside the 44/44 sampler run |
| **S3 PARTIAL — combined authority line absent** | **CLOSED** | `GATE-12 SATURATION VERDICT` line added above both sub-verdicts, computed as the conjunction; both sub-verdicts retained unchanged; F1 required-element list extended 17 → 19. §3 |
| **S4 OPEN** | **CLOSED** | new `active_workers_json` column persisted in every sample, keyed off `obs_status`, invariant enforced at the render site. §4 |
| **E1 OPEN** | **CLOSED** | `tests/test_s172_resolved_execution_set.py` **34/34 GREEN**, expectations derived from the authoritative fixture, derivation demonstrated by perturbation. §5 |

**Three conflicts are reported rather than resolved silently — see §9. One of them
(gate 22) leaves `test_s172_phase4_coordinator.py` at 62/63 and needs a Beta or owner
decision; it is bookkeeping, not behaviour, and the analysis is complete.**

---

## 2. P2 — Gate-12-only GPU fail-close

**Falsifiable question:** does the harness refuse to launch unless all three rigs truthfully
report `status == OK` and the full device count? **Yes.**

### 2.1 Placement in the launch flow

New step **0.5**, in `gate12_launch.sh`, between the evidence block and the clean slate:

```
0.   evidence block ($EVID opened)            line  51
0.5  GPU FAIL-CLOSE GATE   <-- NEW            line  86   (refusal: exit 1)
1.   clean slate (pkill, config move)         line  95
2.   concurrency sampler armed                line 114
3.   coordinator / WATCHER                    line 127
5.   fleet launch                             line 151
```

Beta required the refusal to abort *before the sampler starts and before any coordinator
process is created*. It aborts **earlier than that** — before the clean slate too — so a
refused attempt leaves the box exactly as it found it: nothing killed, no
`optimal_window_config.json` moved, no process spawned. A refused attempt therefore costs
nothing and is retryable once the rigs are up. The refusal is written to the evidence block
(`| tee -a "$EVID"`), which is already open at that point.

Ordering is asserted structurally by `P2-REFUSAL-PRECEDES-SAMPLER`, which reads the live
script and requires `gate < clean-slate < sampler < coordinator < fleet`:
`gate@86 < clean-slate@95 < sampler@114 < coordinator@127 < fleet@151`.

### 2.2 The probe is the certified one, not a second implementation

`scripts/gate12_gpu_gate.py` imports `_build_gpu_probe_script` and `_parse_gpu_probe` from
`preflight_check` and re-exports them by identity. It contains **no probe string and no
parser of its own** — a second implementation would be a second place for the `|| echo 0`
class of defect to live. Asserted on the objects, not on an import line:

```
P2-REUSES-CERTIFIED-PROBE   G._build_gpu_probe_script is PF._build_gpu_probe_script
                            G._parse_gpu_probe        is PF._parse_gpu_probe
```

**`preflight_check.py` is byte-unchanged** (`cfbde94c…`). The generic `PreflightChecker`
advisory policy is untouched: GPU findings there are still `add_warning`, still
non-blocking, and the WATCHER gate that enforces that is still green. This is a
harness rule, exactly as Beta ordered.

### 2.3 Targets and expected counts are derived, not hardcoded

Per CLAUDE.md §1.4. `gate_targets()` resolves the committed execution set
(`rig_profiles_config.json` joined with `distributed_config.json`) and takes `remote_nodes()`:

```
rrig6600   192.168.3.122  expected 8
rrig6600b  192.168.3.156  expected 8
rrig6600c  192.168.3.164  expected 8
```

These are the three addresses Beta enumerated. `P2-TARGETS-ARE-DERIVED` also asserts that
**no rig address literal appears in the gate's executable code**. A gate probing addresses
the run does not use would be worse than no gate — it would pass while the fleet was elsewhere.

### 2.4 The full input space, with observed behaviour per row

Every row was executed, not reasoned about. No real rig was contacted: an `ssh` shim on
`PATH` runs the probe string through a real shell against a controlled fixture `PATH` — the
same technique as `tests/test_preflight_gpu_probe.py`. `scripts/gate12_gpu_gate.py` runs
**unmodified**; the fixture replaces the transport, never the code under test. A per-host
dispatch shim makes exactly **one of the three** rigs misbehave, which is the real case —
the fleet is not uniformly broken.

| probe result on ANY rig | required | **observed** | arm |
|---|---|---|---|
| `OK, count == 8` on all three | proceed | **rc=0, "PASS — 3/3 rigs OK at full count"** | `P2-8x8x3-ALLOWED` |
| `UNAVAILABLE` — ssh failure (`ssh_exit_255`) | REFUSE | **rc=1, rig named, reported UNAVAILABLE** | `P2-UNAVAILABLE-REFUSES` |
| `UNAVAILABLE` — no binary (`binary_not_found`) | REFUSE | **rc=1, rig named, reported UNAVAILABLE** | `P2-UNAVAILABLE-REFUSES` |
| `UNAVAILABLE` — non-zero exit (`rocm_smi_exit_3`) | REFUSE | **rc=1, rig named, reported UNAVAILABLE** | `P2-UNAVAILABLE-REFUSES` |
| `UNAVAILABLE` — timeout | REFUSE | **rc=1, rig named, reported UNAVAILABLE** | `P2-UNAVAILABLE-REFUSES` |
| `ERROR` (unparseable count) | REFUSE | **rc=1, rig named, ERROR, no invented count** | `P2-ERROR-REFUSES` |
| `OK, count == 0` (the genuine observed zero) | REFUSE | **rc=1, and reported as a real `0/8`, status OK** | `P2-COUNT-MISMATCH-REFUSES` |
| `OK, count == 7` | REFUSE | **rc=1, reported `7/8`** | `P2-COUNT-MISMATCH-REFUSES` |

All four UNAVAILABLE causes are enumerated rather than sampled: `ssh_exit_255` and
`binary_not_found` reach the refusal by *different paths* inside the gate (transport vs
classifier), so testing one would leave the other arm untested.

**The honesty requirement holds in both directions, and both are asserted:**

- an UNAVAILABLE or ERROR rig **never renders count-shaped** — the arms assert `"0/8"` and
  `"None/8"` are absent from the output. Refusal text reads
  `count=UNAVAILABLE (expected 8) — reason=ssh_exit_255`.
- a **genuine** zero renders as a genuine `0/8` with status `OK`. That is the opposite
  requirement, and conflating the two is the defect the three-outcome probe exists to prevent.

`evaluate()` fails closed by construction: a status it does not recognise falls through to
refusal, not to proceed.

### 2.5 A real defect found and fixed in the wiring

The first wiring used `if ! python3 scripts/gate12_gpu_gate.py | tee -a "$EVID"`. **A
pipeline exits with the status of its LAST command — `tee` — which is 0 essentially always.**
That form prints `REFUSED` and launches anyway: a decorative gate, in the very script whose
attempt-1 defect was a GPU reading that stopped nothing. The live script reads
`${PIPESTATUS[0]}`. Proven by *executing* both forms against a stub that exits 1, not by
reading text:

```
P2-MUTANT-PIPESTATUS-BYPASS   `| tee` form  -> "LAUNCHED"   (refusal swallowed)
                              PIPESTATUS    -> "ABORTED"
                              and the live script uses PIPESTATUS[0]
```

---

## 3. S3 — the authoritative combined saturation verdict line

**No algorithmic change.** `exit_code()` already encoded the semantics; the gap was that an
exit status is consumed by a process, not by a reader, and `gate12_launch.sh` never reads it.
New `overall_satisfied(v)` returns `v["satisfied"] and v["turnover_satisfied"]`.

### 3.1 Verbatim summary sample — authority line above both sub-verdicts

Simultaneity satisfied, turnover NOT (the case the line exists to catch):

```
GATE-12 SATURATION VERDICT                : NOT SATISFIED
  ^ THE AUTHORITATIVE LINE. It is the CONJUNCTION of the two verdicts
    below (criterion 1 AND criterion 2); both must be SATISFIED. The
    two sub-verdicts are retained underneath as diagnostics — they say
    WHICH criterion failed — but this line is the Gate-12 result.
VERDICT 1 — SUSTAINED SIMULTANEITY        : SATISFIED
VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY : NOT SATISFIED
EXIT CODE                                 : 3
  0 = both criteria satisfied · 2 = criterion 1 (simultaneity) NOT satisfied · 3 = ...
```

Both satisfied:

```
GATE-12 SATURATION VERDICT                : SATISFIED
  ^ THE AUTHORITATIVE LINE. ...
VERDICT 1 — SUSTAINED SIMULTANEITY        : SATISFIED
VERDICT 2 — TURNOVER UNDER FULL OCCUPANCY : SATISFIED
```

Both diagnostic sub-verdicts are kept underneath, **unchanged**, and the exit-code legend is
untouched. Verdict semantics, `evaluate`, and the exit codes 0/2/3 are unchanged.

### 3.2 Gates

Both drive the **real `evaluate` + `render_summary`**, never a hand-built dict:

- `S3-AUTHORITY-LINE-IS-CONJUNCTION` — `yes/yes → SATISFIED`; `yes/no → NOT SATISFIED`; both
  sub-verdicts present; and the authority line's index precedes both sub-verdicts' indices.
- `S3-MUTANT-OR-COLLAPSES-VERDICT` — with `or` substituted for `and`, a turnover-failed run
  publishes itself as `SATISFIED`. **Red as required.**

### 3.3 The extended F1 element list

`f1_summary_is_self_describing` now requires **19** elements, up from 17. Two added:

```
"combined authority":           S.LABEL_OVERALL
"authority is the conjunction": "CONJUNCTION of the two verdicts"
```

Beta's point exactly: without them a mutant deleting the line still passes the 17-element
check. Demonstrated — with the authority line deleted from `render_summary`:

```
[FAIL] F1-SUMMARY-IS-SELF-DESCRIBING    MISSING: ['combined authority']
```

and `S3-AUTHORITY-LINE-IS-CONJUNCTION` reds too. The delete mutant is caught in two places.

---

## 4. S4 — durable simultaneous worker identities in the TSV

**Falsifiable question:** after the process exits, can the evidence file prove *which* 25
workers were simultaneously active? **Yes.**

### 4.1 Column definition

New column **`active_workers_json`**, placed immediately after `compute_active`, where its
invariant binds. Encoding stated at the column definition: **a JSON array of worker-id
strings, SORTED, serialized with `separators=(",",":")`** — no whitespace, no literal tab, so
the field is byte-deterministic for a given set and a reader parses it with `json.loads`. No
whitespace-dependent splitting is required or implied. Sortedness is what lets the evidence
file be diffed or hashed as evidence.

`format_tsv_row` was restructured to key cells **by column name** and emit them in
`TSV_COLUMNS` order, so a column added to the header without a value now raises `KeyError` at
the first write instead of silently shifting every field to its right by one.

### 4.2 The three rendering cases — TSV excerpts

Rendering is keyed off **`obs_status`, not off the set's contents**. The trap Beta flagged is
real and was read in source first: `unobserved_row:311` seeds `active_workers: set()`, so a
renderer serializing the set unconditionally emits `[]` on exactly the samples where `[]` is a lie.

```
1. OBSERVED, 25 simultaneous workers
   obs_status=OBSERVED   compute_active=25
   active_workers_json=["rrig6600:gpu0","rrig6600:gpu1",...,"rrig6600c:gpu7","zeus:gpu0"]
                        (25 ids, sorted, auditable against the frozen execution cohort)

2. UNOBSERVED (failed ledger read)
   obs_status=UNOBSERVED compute_active=UNOBSERVED
   active_workers_json=UNOBSERVED          <-- the marker, NEVER []

3. OBSERVED ZERO (ledger read succeeded, no run of ours yet)
   obs_status=OBSERVED   compute_active=0
   active_workers_json=[]                  <-- a real observation of nothing
```

Cases 2 and 3 being distinguishable **is the point**: `[]` is a positive claim that an instant
was observed and no worker was active; a failed read observed nothing at all.

### 4.3 The invariant and its enforcement site

`len(parsed array) == compute_active`, raised as `AssertionError` **inside
`render_active_workers`** — the last point at which the count and the identities are both in
hand, and the point Beta specified (where the row is rendered). If they disagree the count is
not auditable and the row must not reach the evidence file.

### 4.4 Gates

- `S4-IDENTITIES-PERSIST-GAPS-ARE-NOT-EMPTY` — **end to end through the real `main()`**, real
  loop, real TSV writer. 7 observed rows each carry 25 sorted ids with `len == compute_active`
  **verified from the persisted bytes**, not in-process; 2 injected gaps carry `UNOBSERVED`
  and never `[]`.
- `S4-OBSERVED-ZERO-IS-A-REAL-EMPTY-ARRAY` — end to end, no run latched: 9 rows,
  `obs_status=OBSERVED`, `active_workers_json=[]`, `compute_active=0`.
- `S4-INVARIANT-ENFORCED-AT-RENDER` — 2 identities vs `compute_active=25` is refused; the
  consistent control renders.
- `S4-MUTANT-UNCONDITIONAL-SERIALIZATION` — the naive renderer emits `[]` for a failed read;
  the `obs_status`-keyed one emits `UNOBSERVED`. **Red as required.**

`evaluate` and summary semantics are unchanged — this is persistence, not a new criterion term.

---

## 5. E1 — the stale certified execution-set test

**Two sites were stale, not one.** The brief names only `:667`. A full run found a second
identical staleness at `:649`. Both are the same root cause — the certified localhost GPU
count correction 2 → 1 — and the deliverable is *the full suite green*, so both were fixed
with the same derivation principle. Reported in §9.2.

### 5.1 Red-first

```
--- G-PARTIAL-EXPLICIT: membership is never inferred from reachability ---
  line 649: assert len(s.nodes) == 4 and s.gpu_count() == 26     AssertionError
--- G-CONSUMERS 2/6: legacy test_connectivity reads the set ---
  line 667: assert len(workers) == 2 + 8                          AssertionError
RESULT: FAIL   (32/34)
```

### 5.2 The derivations used

Beta preferred deriving from the authoritative fixture over a fresh magic `1 + 8`.

**`:667`** — `s.gpu_count()` (`execution_set.py:220-221`, `sum(n.gpu_count for n in self.nodes)`)
for exactly the nodes the arm declares. Non-vacuous: the workers come from
`c.create_gpu_workers()` on the coordinator, the expectation from the resolved set.

```python
expected_workers = s.gpu_count()
assert len(workers) == expected_workers, (...)
```

**`:649`** — that arm asserts `s.gpu_count()` *itself*, so deriving from `s.gpu_count()` would
be vacuous. Derived instead from `distributed_config.json` via a new `_config_gpu_total()`
helper — the same file `resolve_execution_set` reads at `execution_set.py:640`
(`int(cfg.get("gpu_count", 0))`), summed independently in the test.

The arm's return strings carry the derived numbers rather than baking counts; no other
assertion in either arm referenced a count. **No production execution-set change.**

### 5.3 Full-suite green, with the counts shown derived

```
[PASS] G-PARTIAL-EXPLICIT: membership is never inferred from reachability —
       resolution performs no reachability probe; down nodes still resolve
       (4 nodes, 25 GPUs derived from distributed_config.json)
[PASS] G-CONSUMERS 2/6: legacy test_connectivity reads the set —
       MultiGPUCoordinator nodes=['localhost', '192.168.3.122'],
       9 GPU workers (derived from set gpu_count=9)

34/34 resolved-execution-set checks green
RESULT: PASS
```

### 5.4 Perturbation proof — the expectation moves

`distributed_config.json` localhost `gpu_count` temporarily `1 → 3`:

```
[PASS] ... down nodes still resolve (4 nodes, 27 GPUs derived from ...)   [25 -> 27]
[PASS] ... 11 GPU workers (derived from set gpu_count=11)                 [ 9 -> 11]
34/34 checks green
```

Both expectations tracked the fixture; a magic literal would have gone red. **Restored from a
byte-exact backup and verified:** `sha256 ac4ba07c…` before and after, `git status` clean for
that file.

---

## 6. Red-first / mutation evidence per new arm

### 6.1 Red-first: every new arm run against the pre-R3 sampler (`49ff9b4`)

The suite is deliberately fail-loud, so one `AttributeError` aborts before later arms run. A
driver caught per arm so each new gate's red is demonstrated **individually**, not inferred
from one abort:

```
[RED AttributeError] s3_authority_line_is_the_conjunction: no attribute 'LABEL_OVERALL'
[RED AttributeError] s3_mutant_or_instead_of_and:          no attribute 'overall_satisfied'
[RED KeyError      ] s4_identities_persist_and_gaps_are_not_empty: 'active_workers_json'
[RED KeyError      ] s4_observed_zero_is_a_real_empty_array:       'active_workers_json'
[RED check-failed  ] s4_invariant_is_enforced_at_the_render_site
[RED ValueError    ] s4_mutant_unconditional_serialization: 'active_workers_json' not in list
[RED AttributeError] f1_summary_is_self_describing:        no attribute 'LABEL_OVERALL'

7/7 arms RED against the pre-R3 sampler
```

P2's arms are red-first by construction — `scripts/gate12_gpu_gate.py` does not exist at
`49ff9b4`. The **bypass mutant** (gate present but not wired into the launch script, the
script left byte-identical to HEAD `e8e00617…`) reds exactly the two wiring arms:

```
[FAIL] P2-REFUSAL-PRECEDES-SAMPLER    gate@None < clean-slate@64 < sampler@83 < ...
[FAIL] P2-MUTANT-PIPESTATUS-BYPASS
7/9 checks green
```

### 6.2 Mutation table — R3

| mutant | must red | observed |
|---|---|---|
| authority line computed with `or` | S3 arm | **RED** — turnover-failed run publishes SATISFIED |
| authority line deleted from `render_summary` | F1 + S3 | **RED** — `MISSING: ['combined authority']` |
| `active_workers` serialized unconditionally | S4 arm | **RED** — failed read renders `[]` |
| gate verdict discarded (`evaluate` → always allow) | all P2 refusal arms | **RED** — unavailable/error/mismatch all PROCEED (rc 0,0,0) |
| gate not wired into `gate12_launch.sh` | P2 wiring arms | **RED** — `gate@None` |
| `\| tee` instead of `${PIPESTATUS[0]}` | P2 wiring arm | **RED** — refusal swallowed, "LAUNCHED" |

### 6.3 R1 + R2 mutation tables re-run unchanged

All green inside the final full runs, none modified:

```
probe   M1A-MUTANT-AUTHENTIC     `|| echo 0` located in EXECUTABLE code at c4e0037
        M1B-MUTANT-REDS-G2       mutant=0 (0/8) vs fixed=UNAVAILABLE
sampler M1A-LEGACY-OVERSTATES    legacy=19 (overstates by 7 staging) vs fixed=12
        M1B-LEGACY-BLIND-TO-QUEUE legacy selected no pending term; fixed reports 20
exec-set G-MUTANT summary        5 consumer mutants all turned their gate red
```

---

## 7. VERIFICATION — Beta §13, full runs, terminal sentinels

Run sequentially (concurrent S172 runs flake Part B on a free-space race):

| suite | required | result |
|---|---|---|
| `tests/test_preflight_gpu_probe.py` | 12/12 | **12/12 green** |
| `tests/test_gate12_gpu_gate.py` (new, P2) | — | **9/9 green** |
| `tests/test_gate12_concurrency_sampler.py` | 38/38 + S3/S4 arms | **44/44 green** (38 + 6) |
| `tests/test_s172_resolved_execution_set.py` | GREEN (E1) | **34/34 green — RESULT: PASS** |
| `tests/test_seed_domain_cursor_amendment.py` | green | **40/40 green — COMPLETION SENTINEL: PASS** |
| `tests/test_s172_f1_f2_active_lease.py` | 16/16 | **16/16 green** |
| `tests/test_s172_phase4_coordinator.py` | 63/63 | **62/63 — see §9.3** |

Base verification before any edit: HEAD `49ff9b4`, probe **12/12**, sampler **38/38**.

---

## 8. Byte-unchanged confirmation — everything outside the four items

`sha256`, taken at end of session:

```
cfbde94c71b66d07a613b4ef49dbc38088efdb4005d28899e5846c2f2c346730  preflight_check.py          <- UNCHANGED from R1/R2
d21614701c31a7b4509ff8980969ff64a1593e8304b92ef627391b2621911716  execution_set.py
ac4ba07ca3f35d9042521b67a20875a37115f90ddd34a6eb0f71ce2de51e9192  distributed_config.json     <- restored byte-exact after perturbation
c69a26443cf74ec608fe0fb87265589cd1aacd205ef654fd4330212875cd4904  rig_profiles_config.json
a3bf1e41aaf05225b931912f515bed6398ec72060e054419b0c1d287f6f5243d  coordinator.py
fc89a3739e88480a81cf33d4948094ff7472a07ecc8c2252add07107938f561f  persistent_worker_coordinator.py
69956b5a577f4d3dce8a359314da4ca4bd9b2c53e82e0d0e9c1579f459a244c2  miner/range_miner_coordinator.py
0b9a7b86b0cf28858118b9b7c0b4646413e015431c94680520f1d563dc0cc55c  miner/range_miner_worker.py
365c8e3ee9abf80a532900b07e53740af08cd1f71bf7d908dd4db685bccf496d  miner/dataset_authority.py
62849d9f30b1c168875873c06a38b175cf6c9e55c66d916efab93ee498f7bc5e  tests/test_preflight_gpu_probe.py
9a238ad5fa5cd7ed0dda8d4f9a84fd51ef21b3f7cfa7a3104333c85bb252ebce  tests/test_s172_phase4_coordinator.py
91a01220961a4f81732d1caf192187b13e5160814e6e3230525c50d6ded3c38a  tests/test_s172_f1_f2_active_lease.py
2499aa60268320fead618e2496a8a3efbc384d03e953053ff39532565357bd4d  tests/test_seed_domain_cursor_amendment.py
5ca1cccb6911eeb11c841c6c8242657329b69e85a2da3011c8bc02b88ec4185e  scripts/launch_fleet_manual.sh
```

`preflight_check.py` and `tests/test_preflight_gpu_probe.py` match their R2 values exactly
(`cfbde94c…` / `62849d9f…`). F1/F2, coordinator, miner, ledger, lease policy, retry matrix,
seed domain, coverage authority, dataset authority, publication, execution-set production
logic and the generic `PreflightChecker` advisory policy are all untouched.

---

## 9. CONFLICTS — reported, not resolved silently

### 9.1 The brief cites the Beta checklist as both §11 and §10

The brief introduces the checklist as *"copied verbatim per Beta §11"* but the report
instruction says *"The §10 checklist verbatim at the top"*. Same list either way; the section
number differs. **The Beta ruling document itself is not in the repo** — the brief's
transcription is the only copy on the box, so Alpha could not check the numbering against the
source. Closure above is claimed against the transcribed list, which is what the brief
designates as operative.

### 9.2 E1 names one stale site; there are two

The brief states *"The stale expectation is `tests/test_s172_resolved_execution_set.py:667`"*.
A full run shows `:649` (`assert len(s.nodes) == 4 and s.gpu_count() == 26`, in
`g_partial_not_inferred_from_answers`) is stale from the same certified correction. Beta's
E1 deliverable is *the full suite GREEN*, unreachable with `:649` still red, so both were
fixed under the same derivation rule Beta specified. Flagged because it widens the literal
text of the brief by one line, in a different arm from the one named. **No production change.**

### 9.3 Gate 22 vs "touch nothing else" — NEEDS A DECISION

`tests/test_s172_phase4_coordinator.py` is **62/63**. The single red is gate 22, the
coexistence scope-drift detector:

```
AssertionError: unexpected changed .py files: {'scripts/gate12_gpu_gate.py',
  'tests/test_gate12_gpu_gate.py', 'scripts/gate12_concurrency_sampler.py',
  'tests/test_gate12_concurrency_sampler.py'}
```

**Diagnosis.** Gate 22 reads `git status --porcelain` and requires every changed or untracked
`.py` to appear in an explicit allowlist. It names exactly R3's four files and nothing else.
The other 62 gates are green, including every behavioural gate.

- The tree was clean of `.py` changes at `49ff9b4`, so gate 22 was **green at HEAD** — this
  red is caused by R3's own uncommitted work, not by a behavioural regression.
- The allowlist is a **per-deliverable bookkeeping ledger**: its own comments show it being
  extended for F1/F2, Phase-5 D0, and Correction-2 Defect 6, each with a rationale. The
  established convention is that an authorized deliverable declares its files there.
- **No uncommitted deliverable can be green on gate 22 without that entry**, and Alpha cannot
  commit (hard rule 1).

**Proven to be bookkeeping, not behaviour.** In a scratch clone, adding the four files to the
allowlist with an R3 rationale — changing nothing else — restores:

```
63/63 checks green
All checks green — S172 Phase 4 coordinator ... is contract-validated
```

**Alpha did not make that edit in the repo.** `test_s172_phase4_coordinator.py` is not on
Beta's no-touch list, but the brief says *"close all four; touch nothing else"*, and R3 was
scoped explicitly against R2's traceability failure. Editing a certified suite's allowlist is
a scope call for Beta or the owner, not for Alpha to take unilaterally. The file is
byte-unchanged (`9a238ad5…`). Two clean options:

1. **Add the four-line allowlist entry** (restores 63/63 immediately; matches how every prior
   deliverable declared itself; costs one edit outside the four items).
2. **Leave it.** The red self-clears the moment Michael commits — a clean tree yields an empty
   `changed_py`. On that path 63/63 is only observable post-commit.

---

## 10. FILES CHANGED — from `git status`, not recall

```
 M gate12_launch.sh
 M scripts/gate12_concurrency_sampler.py
 M tests/test_gate12_concurrency_sampler.py
 M tests/test_s172_resolved_execution_set.py
?? scripts/gate12_gpu_gate.py
?? tests/test_gate12_gpu_gate.py
```

| file | item | change |
|---|---|---|
| `scripts/gate12_gpu_gate.py` | P2 | **NEW** — the fail-close gate; imports the certified probe, derives its targets |
| `tests/test_gate12_gpu_gate.py` | P2 | **NEW** — 9 gates, full input space, fixture-driven |
| `gate12_launch.sh` | P2 | step 0.5 added ahead of clean slate/sampler/coordinator/fleet; `${PIPESTATUS[0]}` |
| `scripts/gate12_concurrency_sampler.py` | S3 + S4 | `LABEL_OVERALL`, `overall_satisfied`, authority line; `active_workers_json` column, `render_active_workers`, name-keyed `format_tsv_row` |
| `tests/test_gate12_concurrency_sampler.py` | S3 + S4 | 6 new arms; F1 element list 17 → 19 |
| `tests/test_s172_resolved_execution_set.py` | E1 | `_config_gpu_total()` helper; two derived expectations (`:649`, `:667`) |

**Also untracked, not produced by R3** (present before this session, unrelated):
`docs/CLAUDE_CODE_INSTRUCTIONS_PRERUN_R3.md`, `docs/TB_SUBMISSION_PRERUN_R2_AND_RERUN_REQUEST.md`,
`miner_ledger.db-shm`, `miner_ledger.db-wal`, `optimal_window_config.json.stale_1786149572`.
This report adds `docs/CLAUDE_CODE_REPORT_PRERUN_R3.md`.

**The `git add` list is built from this section.** Gate 12 remains HELD; nothing was launched.
