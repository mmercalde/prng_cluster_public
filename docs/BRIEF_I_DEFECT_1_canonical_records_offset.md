# BRIEF-I DEFECT 1 — `utils/canonical_records.py` was not migrated

**Classification: BRIEF-I DEFECT.** Not a pre-existing red. Goes to Beta as such.
**Found by:** read-only offline reproduction of `commit_trial` against the retained
artifacts of run `distributed_config_t1_eed23c7f` at `48a8705`. No fleet, no relaunch.

## 1. The exception, named

```
KeyError: 'offset'
  utils/canonical_records.py:217   in build_mode_records
      "offset":  ctx["offset"],
```

Reached by:

```
AssemblingPhase5Sink.commit_trial      range_miner_npz_writer.py:1268
  _assemble                                                    :1236
    assemble_trial                                             :1118
      merge_validated_spools                                    :949
        _mode_records  (= build_mode_records)                    :836
          build_mode_records                canonical_records.py:217  <-- KeyError
```

This is the exception the production run swallowed into `event["error"]`; it is why
`commit_delivery_status` is `failed`, why `get_assembly()` returned `None`, and why
`MinerIngressError` then failed closed.

## 2. The `_CONTEXT_FIELDS` hypothesis is REFUTED — by evidence, not reasoning

The prior hypothesis was that `range_miner_npz_writer.py:1038`'s 12-field comprehension
`{k: metas[0][k] for k in _CONTEXT_FIELDS}` raised. **It did not.** Measured against the
actual retained manifests:

```
derive_trial_metadata keys   [... generator_phase ... window_anchor ...]
  window_anchor              PRESENT  58
  generator_phase            PRESENT  0
  offset                     ABSENT            (correctly retired from the context)
validate_trial_metadata      OK
publish_shard                5632 / 5632 accepted, no exception
```

The 12-field projection is correct and the seam Brief I built works. **The defect is one
layer further downstream, in a different file, with a second unmigrated consumer of the
same retired key.**

## 3. Why it is Brief I's

Brief I split the trial-context scalar `offset` into `window_anchor` (host) and
`generator_phase` (device) and migrated the coordinator, the worker and the NPZ writer.
**`utils/canonical_records.py` was never opened.**

```
last commit touching the file : 70cd6f0   (S172 Phase 5 D3.25) — NOT 48a8705
not present in Brief I's 20-file change set
:117   CANONICAL_RECORD_FIELDS declares "offset"
:217   build_mode_records reads ctx["offset"]
```

It sits on the **production Phase-5 assembly path** — imported at
`range_miner_npz_writer.py:64`, aliased `_mode_records` at `:836`, called at `:949`. It is
not a Brief-II repo-wide-audit item: it is on the execution path Brief I changed.

## 4. Why 25 green gates did not catch it

`G-PHASE5-SEAM` stops exactly one layer short. It proves the manifest →
`_CONTEXT_FIELDS` projection → shared canonicalizer chain raises no KeyError at
`npz_writer:1026`, and it does that correctly. It never drives `assemble_trial`.

```
occurrences of canonical_records / build_mode_records / assemble_trial
in tests/test_s172_window_anchor_brief_i.py :  0
```

The gate's own docstring says it *"reds on the KeyError at :1026 that C-3(b) predicts if
the two tuples drift."* It anticipated this exact defect **class** and caught the instance
it was aimed at — while a second consumer of the same retired key, in a file the brief
never listed, drifted unobserved. **A gate that proves one consumer migrated is not
evidence that every consumer did.**

## 5. ⚠ THE FIX CANNOT BE A RENAME — `offset` is a FROZEN NPZ ARRAY NAME

```
CANONICAL_ARRAY_CONTRACT index 4 == "offset"        "window_anchor" is NOT in the 22
```

Proposal v1.1 is explicit: *"The 22-array wall STAYS CLOSED. `window_anchor` /
`generator_phase` / `anchor_era` are metadata only — no array added, removed, reordered,
retyped or reshaped."*

So renaming the record field would **breach the frozen 22-array contract**. The record
field must keep the name `offset` while sourcing its VALUE from `ctx["window_anchor"]`.

**That is a semantic statement requiring Beta's ruling, not an implementation detail:** it
declares that frozen array 4, historically the fused scalar, now carries the window anchor
— which is coherent only because v1 pins `generator_phase = 0`. Alpha proposes no patch
and requests a ruling. `CANONICAL_RECORD_FIELDS:117` and `utils/canonical_arrays.py`'s
deliberate duplicate copy are both in scope of that ruling.

## 6. Reproduction was READ-ONLY — retention undisturbed

```
staging file count   6151 before  ->  6151 after
miner_ledger.db      sha fa9b0ecd36464fdf  ->  fa9b0ecd36464fdf   (unchanged)
C2's 512 files       untouched        archived ledger  untouched
```

The repro reconstructs manifests from the ledger through the production
`_trial_context_row_to_ctx` / `derive_trial_metadata` and opens the retained spools for
reading only.

**One repro defect worth recording:** the first attempt copied `miner_ledger.db` without
its `-wal`, so the copy was missing the newest rows and produced a misleading
`TypeError: expected str … not NoneType`. The same WAL lesson as the archive step. Corrected
by reading the live ledger `mode=ro`.
