# TEAM ALPHA → TEAM BETA — PRE-RERUN ITEMS COMPLETE + GATE-12 RERUN REQUESTED

**Both items you required before a Gate-12 rerun request are delivered.** Base **`c4e0037`**
(F1/F2 certified). **Nothing committed, pushed or launched; port 5700 never bound; the launch
script was written but NOT run.** No coordinator, miner, ledger, seed-domain/coverage or certified
suite file was modified — production blast radius is **`preflight_check.py` and
`gate12_launch.sh`, nothing else.**

**Gates: `test_preflight_gpu_probe.py` 12/12 · `test_gate12_concurrency_sampler.py` 14/14** —
both reproduced independently by Alpha on a second host.

**A hash correction first:** Alpha's prior note cited the F1/F2 commit as `d3f8f00`. That hash does
not exist — Alpha misread it from a terminal image. **The correct commit is `c4e0037`**, confirmed
by `git log` and `git cat-file`. Claude Code caught it by attempting the lookup rather than trusting
the brief.

---

## 1. Item 1 — the GPU probe now distinguishes UNAVAILABLE from zero

**Root cause, measured on all three CTs rather than inferred:**

- `/opt/rocm/bin` is placed on PATH by **`~/.bashrc:120` and nothing else** — no
  `/etc/profile.d` script, no `/etc/profile`, no `/etc/environment` entry mentions rocm;
- `~/.bashrc:5-8` is Ubuntu's stock **non-interactive guard**, returning ~112 lines **before** that
  export;
- `bash -l` sources `~/.profile`, which does source `.bashrc` — **but it returns at the guard.**
  **`bash -lc` and a bare non-interactive command observe the byte-identical PATH. Only `bash -lic`
  sees `/opt/rocm/bin`.**

**A login shell was never going to fix this**, which is why the brief required the reason be
established before a remedy was chosen.

**Two constructs each manufactured the zero independently:** the old stdout was `'0\n0\n'` —
`grep -c` printed `0` and exited 1, then `|| echo 0` printed a second one. Also found: `ssh` was
flattening the `["bash","-lc",...]` argv, so the remote shell re-parsed the pipeline with its
quoting already gone — **the semantics survived by luck.**

**The fix:** three distinguishable outcomes — a count, **`UNAVAILABLE`** (`gpu_count: None`, never
`0`), or `ERROR`. The binary is located rather than assumed. `2>/dev/null` and `|| echo 0` are
gone; stderr is captured and surfaced. **Alpha verified the render guard directly** (`:190-200`):
an unavailable node cannot render as `0/8` **or** `None/8`.

**Measured live through the production method: 8/8 on all three rigs** via `/opt/rocm/bin/rocm-smi`.

**Gating is untouched** — `checks_passed += 1  # Don't block on GPU warnings` remains at `:370`,
and five gate arms prove GPU findings stay advisory in every outcome. Red-first: restoring
`|| echo 0` reproduces the exact production string **`GPU_COUNT_MISMATCH: 0/8`** and reds five
gates including the missing-binary arm.

## 2. Item 2 — the sampler measures the post-F1 state model

**Occupancy now mirrors the production authority.** `compute_busy_worker_ids` semantics:
`state='claimed'` only — **staging deliberately excluded, because `StripeComplete` has already
freed the compute slot** — and `pending` is measured as the real F1 backlog. Alpha read the query
directly; the exclusion reasoning is documented inline at the site.

**Both of Alpha's prior defects are closed:** every query is **run-scoped** (the ledger may hold
other runs); connections are `mode=ro` and **production databases are refused by name** (a prior
harness created a real table in the live DB by cwd-relative resolution); the sampler **arms before
the coordinator exists** and **terminates with the run** rather than looping for two hours against
a dead trial.

**The verdict evaluates simultaneity PER SAMPLE.** The max-over-time union is still computed — **so
that it can be printed under a heading stating it cannot qualify.** Alpha verified this in source:
`satisfied` is keyed on consecutive satisfying samples, never on the union. Gate arms prove a
fixture with 25 claimed but **zero pending** does not qualify, and that 25 distinct workers reached
only across different instants does not qualify.

## 3. The launch script — delivered, verified offline, NOT run

`gate12_launch.sh` carries **`worker_pool_size: 25`** and the sampler **moved ahead of the fleet
launch**. Verification was static, not by execution: `bash -n` clean, params matched against the
frozen shape, and a **simulated `EXEC CMD`** confirms `--worker-pool-size 25` and `--trials 1`.

**That simulation caught a real trap:** `trials` is mapped in `args_map`, so the underscore fallback
would have emitted a **non-existent `--window-trials`**. `--use-persistent-workers` is correctly
**absent** rather than passed false.

**Evidence it was not run:** port 5700 unbound, no processes, **ledger mtime unchanged.**

## 4. Two findings reported, neither actioned

1. **A pre-existing failure, not ours.** `test_s172_resolved_execution_set.py` fails at **clean
   `c4e0037`** at the identical line 667 — `f255912` set `localhost.gpu_count` **2 → 1** while the
   gate still asserts `2 + 8`. Stale expectation in a certified suite; **untouched**, reported for
   your disposition.
2. **Phase-4 is 63/63** with `preflight_check.py` modified; the only red was **Gate 22** on the
   three new untracked `.py` files — the documented sensitivity, which clears at commit.

## 5. REQUEST — Gate-12 rerun authorization

Alpha requests authorization for the production-shape rerun. **The shape is unchanged from the one
you authorized**, plus the single correction:

```
seed_start        = 0            (explicit; certified cursor first-gap, empty namespace)
seed_count        = 2,147,483,648  (2^31)  ⇒ 32 macro-stripes per stage
miner_stripe_size = 67,108,864     (2^26)
worker_pool_size  = 25            ← the correction; the manifest default of 8 was never overridden
test_both_modes   = true          prng_type = java_lcg
window_trials     = 1             n_parallel = 1
use_range_miner   = true          use_persistent_workers = false
```

At W=25 the stage opens **25 claimed / 7 pending** — the first time the run will have the real
queue your 32-stripe geometry was chosen to produce.

**Standing conditions Alpha will observe, restated so they are on the record:** no mid-run
intervention of any kind; a sizing refusal at preflight is a legitimate Gate-12 result and will not
be met by shrinking the seed count; coordinator process death means the run is interrupted, not
resumable; GPU completion is not completion — only successful canonical publication is; and if
fewer than 25 workers are admitted, the run will **not** be reported as a saturation pass.

**Nothing will be launched until you authorize it.**
