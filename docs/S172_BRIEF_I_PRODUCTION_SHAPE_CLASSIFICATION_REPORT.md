# BRIEF I — PRODUCTION-SHAPE RUN, CLASSIFICATION REPORT TO TEAM BETA

**Commit under proof:** `48a87059f5200e00727556f05c1462df07ba4614`
**Run:** `distributed_config_t1_eed23c7f` · stamp `20260822_143303` · nonce `gate12-20260822_143303-10569`
**Authorized by:** Michael, 2026-08-22 (`CLAUDE.md` rule 3). **Executed by:** Team Alpha, VM 101.
**Responds to:** `TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` closure §8, steps 2–3.

## VERDICT

| | |
|---|---|
| production-shape proof | **NOT OBTAINED — the run FAILED at Phase-5 commit** |
| Brief I final acceptance | **NOT RECOMMENDED at this commit.** A Brief-I defect is named below |
| cause | **BRIEF-I DEFECT**, reproduced offline and named to the line |
| fleet parity / schema parity (steps 2a/2b) | unchanged — **PASS**, reported separately |
| what Alpha asks of Beta | **four filed items and two ratifications. No patch is proposed and nothing is fixed until Beta rules.** |

---

## 1. WHAT THE RUN PROVED — and it is substantial

The compute path executed end to end at `48a8705` on the full 25-GPU fleet.

```
all four phases          128/128 stripes DONE   (1/2 constant, 3/4 hybrid)
trial state              committed
compute_lease_expiry     0
WORKER_DISCONNECTED      0
pre-terminal ERROR lines 0
serve loop               713.663 s · iterations 3429 · iteration_max 0.796 s
staging                  5632 jobs @ 7.892/s · pause_events 0
                         capacity_timeout_terminations 0 · capacity_invariant_terminations 0
                         inbound_saturation_events 0 · emergency_events_total 0
saturation               qualifying windows found; turnover confirmed under full occupancy
                         (6 drained step-wise, 6 transitions, window 1 of 4)
```

**`iteration_max = 0.796 s` against MP-1's 940.971 s.** The drain-starvation pathology that
defined attempts 6–8 is absent under a real four-phase load. `drain_seconds_per_frame` 0.0587,
`drain_passes_partial` 3412/3429, `pump_total` 0.000 on the serve thread.

**Both `_bp` falsifier fields were persisted and observed** — the §2.52 repair's first
production reading: `deferred_distinct_attempts_high_water=30`,
`pump_liveness_probes_high_water=126`. Neither read `UNOBSERVED`.

### 1.1 The separation itself worked, end to end, across 25 workers

The `trial_context` row this run wrote:

```
window_anchor_val   58          generator_phase   0
window_size         20          sessions          ["evening"]
prng_base           java_lcg    dataset_sha256    513648160d356617…
                                residue_sha256    a21694ac2e584fd0…
```

Optuna sampled `offset=58`; it arrived as **`window_anchor=58` with the phase pinned at 0**,
through the coordinator, over the wire, into 25 deployed Brief-I workers, and back. The retired
`offset_val` column does not exist in the schema. **The thing the brief exists to do, works.**

**All six pre-launch gates passed before a GPU-second:** clean-tree admission · GPU 3/3 at full
count · **rig parity 30/30 at the Brief-I digests** · pre-dispatch clean-tree · sentinel 25/25 ·
liveness 25/25.

---

## 2. THE TERMINAL DEFECT — full call chain

```
KeyError: 'offset'
  utils/canonical_records.py:217   in build_mode_records
      "offset":  ctx["offset"],
```

```
RangeMinerCoordinator.commit_trial              range_miner_coordinator.py:8644-8773
  phase5_sink.commit_trial(event)                                          :8676
    AssemblingPhase5Sink.commit_trial            range_miner_npz_writer.py:1268
      _assemble                                                            :1236
        assemble_trial                                                     :1118
          merge_validated_spools                                            :949
            _mode_records  (= build_mode_records)                           :836
              build_mode_records                  utils/canonical_records.py:217   <-- KeyError
```

**Consequence chain, all of it correct behaviour after the first fault:**

```
commit_delivery_status = failed          (:8680-8686, Option C)
  -> shards RETAINED   5632 verified / phase5_status enqueued / cleanup none
  -> reservations HELD  5632 of 5632
  -> get_assembly() returns None          (never a partial or fabricated result)
  -> MinerIngressError: "no committed assembly … failing closed rather than
     accumulating a fabricated zero-candidate trial"
  -> Step 1 exit 1, no optimal_window_config.json, WATCHER escalation
```

**Option C retention, the fail-closed assembly contract and the ingress wall all behaved
exactly as certified.** Only the first fault is a defect; everything after it is the system
refusing to manufacture a result — and it is what made the offline diagnosis possible at all.

---

## 3. A HYPOTHESIS ALPHA RAISED, TESTED, AND REFUTED

Recorded because a named-and-refuted hypothesis is stronger evidence than one never raised.

**Hypothesis:** Brief I moved `range_miner_npz_writer._CONTEXT_FIELDS` from 11 to 12 fields,
and `:1038` does `ctx = {k: metas[0][k] for k in _CONTEXT_FIELDS}` — a `KeyError` if a manifest
lacks `window_anchor` or `generator_phase`. Alpha proposed this before testing it.

**REFUTED.** Measured against the run's own retained manifests:

```
derive_trial_metadata keys   [ … generator_phase … window_anchor … ]
  window_anchor              PRESENT  58
  generator_phase            PRESENT  0
  offset                     ABSENT           (correctly retired from the context)
validate_trial_metadata      OK
publish_shard                5632 / 5632 accepted — no exception, no replay conflict
```

**The 12-field projection and the seam Brief I built are correct.** The defect is one layer
further down, in a different file, in a *second* consumer of the same retired key that the
brief never opened. Alpha states plainly that its first diagnosis was wrong.

---

## 4. FILED ITEM 1 — BRIEF-I DEFECT: `utils/canonical_records.py` was never migrated

**Classification: BRIEF-I DEFECT. Not a pre-existing red. Not a Brief-II audit item.**

```
last commit touching the file   70cd6f0  (S172 Phase 5 D3.25) — NOT 48a8705
present in Brief I's 20-file change set : NO
  :117   CANONICAL_RECORD_FIELDS declares "offset"
  :217   build_mode_records reads ctx["offset"]
```

Brief I split the context scalar and migrated the coordinator, the worker and the NPZ writer.
This consumer sits on the **production Phase-5 assembly path** — imported at
`range_miner_npz_writer.py:64`, aliased `_mode_records` at `:836`, called at `:949`. It is on the
execution path Brief I changed, so it belonged in Brief I's scope, not Brief II's repo-wide
consumer audit.

**Alpha proposes no patch.** See §5 — the obvious fix is barred by a frozen contract.

---

## 5. FILED ITEM 2 — A RULING IS REQUESTED, NOT A PATCH: the frozen array name

**The fix cannot be a rename.**

```
CANONICAL_ARRAY_CONTRACT index 4 == "offset"
"window_anchor" is NOT among the 22
```

Proposal v1.1: *"The 22-array wall STAYS CLOSED. `window_anchor` / `generator_phase` /
`anchor_era` are metadata only — no array added, removed, reordered, retyped or reshaped."*
Renaming the record field would breach the frozen contract; leaving it unfed leaves the defect.

**The only shape that satisfies both is a semantic declaration, and it is Beta's to make:**

> **Frozen array 4 keeps the name `offset` and sources its value from `ctx["window_anchor"]`.**

That is coherent **only while v1 pins `generator_phase = 0`** — with the phase at zero the
anchor is the whole of what the historical fused scalar meant, so the array's values remain
comparable in kind. It stops being coherent the moment a nonzero phase is permitted.

**Alpha therefore asks Beta explicitly:**

1. Is that declaration approved as the disposition of array 4 for v1?
2. **What happens to that coherence at ABI-v2?** `DEP-ABI-V2` is already recorded (§2.53) as
   the separate kernel/parity certification cycle for independent phase on the four no-phase
   forward hybrids. If a nonzero `generator_phase` ever becomes permissible, array 4 would
   carry an anchor while a second, unrecorded quantity moved independently — the exact
   fusion F-4 exists to eliminate, reintroduced at the NPZ boundary rather than the kernel one.
   Does the 22-array wall then need a governed amendment, and should that dependency be
   recorded against `DEP-ABI-V2` now rather than discovered there?
3. `CANONICAL_RECORD_FIELDS:117` and `utils/canonical_arrays.py`'s **deliberate duplicate**
   copy of the field list (its `:130-142` note: the duplication is intentional and
   gate-checked) are both inside the scope of whatever is ruled.

**No code will be written against this until Beta rules.**

---

## 6. FILED ITEM 3 — GATE-COVERAGE FINDING: the seam gate stops one layer short

`G-PHASE5-SEAM` is not defective. It proves the manifest → `_CONTEXT_FIELDS` projection →
shared canonicalizer chain raises no `KeyError` at `npz_writer:1026`, and it proves it
correctly. **It never drives `assemble_trial`.**

```
occurrences of canonical_records | build_mode_records | assemble_trial
across tests/test_s172_window_anchor_brief_i.py (25 gates) :  0
```

The gate's own docstring reads: *"Reds on the KeyError at range_miner_npz_writer.py:1026 that
C-3(b) predicts if the two tuples drift."* **It anticipated this defect class exactly, and
caught the instance it was pointed at** — while a second consumer of the same retired key, one
layer downstream in a file the brief never listed, drifted unobserved.

**The durable lesson, in the project's own terms (§2.44):** a gate that proves *one* consumer
migrated is not evidence that *every* consumer did. The coverage boundary was the brief's file
list, and the brief's file list was the thing that was wrong.

**Alpha asks Beta, rather than building it:** must the seam gate's scope extend through
`assemble_trial` to the record builder — i.e. should the certifying property be *"a real
trial context survives the whole publish → assemble → record-build path"* rather than *"the
projection is well-formed"*? That is a widening of a certified gate's contract and Alpha will
not widen it unilaterally.

---

## 7. FILED ITEM 4 — OBSERVABILITY GAP: `commit_trial`'s exception never reaches the log

```python
except Exception as e:                      # range_miner_coordinator.py:8680
    self.ledger.set_trial_commit_status(run_id, "failed")
    event["error"] = str(e)                 # :8681  -> IN-MEMORY DICT ONLY
```

`event` is a local dict. No logger call on this path; the ledger persists only the three-state
`commit_delivery_status`, never the reason.

```
Phase-5 / commit / assembly log lines in the run    0
commit_delivery_status                              failed
the exception text                                  NOWHERE
```

A terminal failure that ended a 25-GPU four-phase run left **no record of its own cause**. The
operator saw only a downstream `MinerIngressError` naming the symptom.

**Third instance of a named class:** F2's `_handle_stripe_failure_locked` building a precise
reason and emitting nothing (§2.26); `_conn_reader_loop`'s nine exits funnelling into one bare
`eof` (§2.39). The neighbouring capacity-timeout path **in this same file** does `logger.error`
first (`:6031-6032`) — an inconsistency inside one file, exactly as F2 was.

**Recorded, not repaired.** Any repair is a coordinator change needing its own authorization and
must **not** be folded into the §5 fix.

---

## 8. B7 — CLASSIFIED

**SAME PRE-EXISTING RED, SAME MECHANISM.**

```
this run       0 / 5632 non-'none'
nine historical runs   0 / 29,082 non-'none'
guard          _delete_remote's call site: `if task.kind == "remote":`  (:4527)
               never true, on a run where 24 of 25 workers are remote
```

Against Beta's four accept-while-red conditions:

| condition | status |
|---|---|
| same mechanism | ✅ identical guard, identical 0-of-N |
| no Brief-I definition causally involved | ✅ zero occurrences of `_delete_remote`, `_finalize_stage`, `set_remote_delete`, `remote_delete_status`, `retry_remote_delete` in Brief I's diff |
| **Brief-I legs pass** | ⛔ **CANNOT BE CERTIFIED** — the run did not publish, so no Brief-I publication leg was exercised |
| no new regression | ✅ on this leg |

**Alpha does not claim B7 satisfies the accept-while-red test.** Three of four hold; the third
is unobtainable until a run publishes.

---

## 9. B5 / B6 / B8 — A REAL DIFFERENCE FROM ATTEMPT 9, STATED AS ONE

| leg | attempt 9 (`e9ca800`), run-scoped | this run (`48a8705`), run-scoped |
|---|---|---|
| B5 phase-5 acked | **5693 / 5693 PASS** | **0 / 5632 FAIL** |
| B6 local cleanup | **5693 / 5693 PASS** | **0 / 5632 FAIL** |
| B8 reservations held | 0 held | **5632 held of 5632** |

**These are not pre-existing reds and are not presented as such.** They are downstream of the
single Phase-5 commit failure — Option C retains everything when delivery fails, which is the
certified contract — but the outcome genuinely changed relative to attempt 9 and is reported as
a difference, not absorbed into the historical picture.

---

## 10. TWO OPERATIONAL DEVIATIONS SUBMITTED FOR RATIFICATION

Both owner-authorized under §7, both disclosed rather than assumed.

**10.1 Pre-separation ledger displacement** —
`PROD_SCOPE_1_EXCEPTION_LEDGER_DISPLACEMENT.md`. Required because G-MIGRATE deliberately does
not migrate, making the legacy ledger permanently unusable by the post-separation engine, while
the production shape pins the ledger path. Executed in Beta's order: sha256 → `cp` outside the
staging namespace → **re-hash and require byte-identity** → functional read (9 run_ids, 768
stripes, 29,082 shards) → **only then** `mv` the originals aside. `-wal` displaced with the main
file, since a fresh ledger inheriting a stale WAL is a corruption risk. **Nothing deleted; two
byte-identical copies exist.** The archive remains the queryable evidence base for the B7
investigation and the C2 correlation.

**10.2 Ledger pre-creation** — `LEDGER_BIRTH_PROVENANCE.json`. The harness arms the concurrency
sampler **before** the coordinator, and the sampler refuses when the ledger is absent (*"The
sampler never creates a database"* — a sound anti-fabrication guard). The coordinator is what
creates the ledger. On a fresh ledger the two requirements are mutually unsatisfiable. The
ledger was therefore created through the **production `MinerLedger` constructor**, and its birth
provenance captured **before launch**:

```
sha256 at birth   74c0b150f3b7c538951ea08fc1963b5fa40a1d561879915dd78b5a65a11ff209
size              94,208 bytes      tables 7      TOTAL ROWS 0
window_anchor_val present · generator_phase present · offset_val ABSENT
```

**Every row in the run's ledger was therefore written by the run itself.** No anti-fabrication
leg of `G-PROD-SHAPE` concerns ledger creation time (A1/A3 test manifest-origin of
`staging_dir`; A4 the absent alias; A5 the miner flag; A6 the production validator; A7 no
substitute coordinator; A8 the pre-repair failure).

**10.3 A latent harness defect, filed:** the launch harness cannot start against a fresh ledger.
It has only ever run where one already existed — continuously since before Gate-12 attempt 1
(2026-08-08). **It will recur on every fresh deployment or legitimate ledger reset.** This is the
§2.30 pattern: the harness passes because it encodes the same assumption the implementation
does. Recorded, not repaired.

---

## 11. EVIDENCE

```
prodshape_48a8705_RAW.log                     G-PROD-SHAPE UNCHANGED, raw: 15 pass / 6 fail
prodshape_runscoped_48a8705.log / .json       run-scoped probe: 6 pass / 5 fail,
                                              512 historical files excluded from the verdict
prodshape_runscoped_probe.py                  the probe, as executed (read-only)
BRIEF_I_DEFECT_1_canonical_records_offset.md  the defect, call chain, refuted hypothesis
OBSERVABILITY_GAP_1_commit_trial_exception.md the gap
PROD_SCOPE_1_EXCEPTION_LEDGER_DISPLACEMENT.md deviation 10.1
LEDGER_ARCHIVE_MANIFEST.json                  hash, size, mtime, 9 run_ids w/ per-run counts
LEDGER_BIRTH_PROVENANCE.json                  empty-at-birth proof, full schema + DDL
B7_remote_delete_status_source_check.md        B7 source check
C2_orphan_staged_payloads.json                 512 files, sha256/size/mtime — untouched
parity_gate_48a8705.log/.json                  30/30 at the Brief-I digests
schema_probe_48a8705.jsonl                     3 rigs, 0 failed checks
S172_..._CLOSURE_STEP23_REPORT.md              steps 2a/2b
logs/gate12_20260822_143303*                   the run's own artifacts
```

## 12. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every command wrapped `; echo "EXIT=$? SENTINEL_DONE"`; both probes and the repro carry their sentinels.
- **clean control:** ledger empty at birth (0 rows, §10.2) — the run's rows cannot be inherited.
- **fault-injection control:** not applicable; this is a production observation, not a gate.
- **completion sentinel:** G-PROD-SHAPE `FAIL`; run-scoped probe `FAIL`; repro reproduced the exception deterministically.
- **unobservable ≠ clean:** the commit exception is reported **UNAVAILABLE from the log** and was obtained only by offline reconstruction; the GCVM_L2 criterion remains `UNAVAILABLE`, never `PASS`.
- **audit claim scope:** one production-shaped trial at `48a8705`. **No claim of a passing production-shape proof. No Attempt-9 or soak evidence is used as substitute.**
- **searched surfaces:** live tree at `48a8705` · the run's log, ledger and retained spools · AST of coordinator/worker/npz_writer/canonical_records · `git diff e9ca800..48a8705` · the Brief-I gate suite.
- **unavailable surfaces:** the commit exception in any durable artifact (§7); rig GPU kernel logs (unprivileged LXC).
- **governance trail searched:** the Brief-I code-review and scope rulings · proposal v1.1 · skill §2.44/§2.52/§2.53/§2.54.

**One repro defect of Alpha's own, recorded:** the first offline attempt copied the ledger
without its `-wal` and produced a misleading `TypeError` — the same WAL lesson as §10.1.
Corrected by reading the live ledger `mode=ro`.

## 13. STATE AT HANDOVER

```
HEAD                 48a87059f5200e00727556f05c1462df07ba4614
tracked tree         CLEAN — 0 entries
committed / pushed   NOTHING
fleet                0 workers on all three rigs; port 5700 unbound
retention            5632 shards + manifests, C2's 512 files, archived ledger — ALL UNTOUCHED
                     staging 6151 files before repro -> 6151 after; ledger sha unchanged
```

**Nothing is fixed until Beta rules.**
