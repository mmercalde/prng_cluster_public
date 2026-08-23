# OBSERVABILITY GAP 1 — `commit_trial`'s exception never reaches the log

**Filed as its own investigation item.** Same class as the F2 failures: causal information
destroyed before any durable observer receives it.

## The gap

`range_miner_coordinator.commit_trial` (`:8644-8773`):

```python
except Exception as e:                      # :8680
    self.ledger.set_trial_commit_status(run_id, "failed")
    event["delivery"] = "failed"
    event["error"] = str(e)                 # :8681  -> IN-MEMORY DICT ONLY
```

`event` is a local dict. **No `logger` call on this path, and the ledger persists only the
three-state `commit_delivery_status`, never the reason.** Measured on run
`distributed_config_t1_eed23c7f`:

```
grep -c "phase5|Phase-5|commit_trial|publish_shard|assembly"  ->  0 lines
commit_delivery_status                                        ->  failed
the exception text                                            ->  NOWHERE
```

## Why it matters, concretely

A terminal failure that destroyed a full 25-GPU four-phase run left **no record of its own
cause**. The operator saw only a downstream `MinerIngressError` about a missing assembly —
which names the symptom, not the defect. Naming the actual `KeyError: 'offset'` required
reconstructing the manifests from the ledger and re-driving the sink offline.

That reconstruction was only possible **because Option C retention worked**. Had the commit
path released its artifacts, the defect would have been unreproducible and the run
undiagnosable.

## Precedent

This is the third instance of the class the project has already named:
`_handle_stripe_failure_locked` building a precise reason string and emitting no log record
(F2, §2.26); `_conn_reader_loop`'s nine exits funnelling into one bare `eof` with no reason
(§2.39). Both were repaired by making the reason durable. **The neighbouring
capacity-timeout path in this very file does `logger.error` first** (`:6031-6032`), so this
is an inconsistency inside one file, exactly as F2 was.

## Scope

Recorded, not repaired. Any repair is a coordinator change and needs its own authorization;
it is **not** to be folded into the Brief-I defect fix.
