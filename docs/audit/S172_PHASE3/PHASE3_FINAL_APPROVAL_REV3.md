---
Status: Approved
Phase: S172 Phase 3
Applies to: range_miner_worker.py
Supersedes: PHASE3_FIX_BRIEF_REV2.md
Final validation: 14/14 gates green on RTX 3080 Ti (VM 101); ROCm deploy validation deferred to Phase 6
---

# S172 Phase 3 — FIX BRIEF v3 (Team Beta rev-2 rejection → rev-3)

**For:** Claude Code on VM 101 as `michael`, `/home/michael/distributed_prng_analysis`.
**Status:** rev-2 REJECTED. Two implementation blockers remain + four test
corrections. Beta's ruling is binding. This is a TIGHT scope — do NOT touch the
parts Beta approved.

**Beta APPROVED and must stay unchanged:** the six forward-hybrid ABIs, the
reverse-HYBRID builders (14 args), the content-keyed `ResidueResolver`, the
canonical `s172_substripe_v1` spool format, `supported_variants()` validation +
`VariantStopCondition`, and the `try/finally` `_best_effort_gpu_cleanup` structure.

Only three code areas change: reverse-CONSTANT builders, spool selection guard,
and four gates.

---

## Blocker 1 — reverse CONSTANT builders have the wrong ABI (CORRECTNESS)

**Verified against live `prng_registry.py`** (`lcg32_reverse_sieve` :2380,
`xorshift32_reverse` :2531, `java_lcg_reverse` :3117, `minstd_reverse` :3247,
and pcg32/xorshift128 reverse): every fixed-skip REVERSE kernel has the signature
```
candidate_seeds, residues, survivors, match_rates, best_skips, survivor_count,
int n_candidates, int k, int skip_min, int skip_max, float threshold, int offset
```
That is **`_constant_prefix` + `int32(offset)` = 12 args**. The generator params
(a, c, m, shifts, increment) are **hardcoded inside the reverse kernel body** —
they are NOT passed. (Same pattern the reverse HYBRIDS already use correctly.)

rev-2 wrongly appends the forward family tail before offset, producing 13–15 args.
Wrong for all six reverse constant variants.

**Fix — in EVERY constant builder, split the reverse branch:**
```python
def build_<fam>(ctx):
    if ctx.hybrid:
        ...   # unchanged — Beta approved
    if ctx.reverse:
        # reverse constant kernels hardcode generator params in-kernel:
        # _constant_prefix + int32(offset) = 12 args (registry-verified)
        return _constant_prefix(ctx) + _offset_tail(ctx)
    # forward constant: keep the family-specific tail + offset (unchanged)
    return _constant_prefix(ctx) + [<family tail>] + _offset_tail(ctx)
```
Apply to all six: java_lcg, lcg32, minstd, pcg32, xorshift32, xorshift128. The
FORWARD constant branches keep their family-specific params exactly as-is.

**Do NOT** collapse forward and reverse — only the reverse constant path loses the
family tail. Forward constant and forward hybrid are unchanged.

---

## Blocker 2 — oversized inline candidate throws before the spool branch (DATA/STABILITY)

**Verified:** `message_to_bytes` → `encode_frame` raises `ValueError` at
`range_miner_protocol.py:219-220` as soon as the body exceeds `MAX_FRAME_BYTES`
(64 MiB). rev-2 does `if len(message_to_bytes(candidate)) <= INLINE_BYTE_LIMIT`,
so a candidate over 64 MiB raises INSIDE the guard and never reaches spool — the
exact frame-overflow failure blocker 2 was meant to remove.

**Fix — decide to spool WITHOUT framing a known-large candidate:**
```python
# Payload alone already at/over the inline ceiling → spool, don't frame it.
should_spool = len(payload_bytes) >= INLINE_BYTE_LIMIT
if not should_spool:
    try:
        should_spool = len(message_to_bytes(candidate)) > INLINE_BYTE_LIMIT
    except ValueError:
        # Protocol hard-cap exceeded while framing → must spool.
        should_spool = True
if not should_spool:
    return candidate
# else: spool the exact payload_bytes atomically, clear inline (unchanged path)
```
Keep the 48 MiB `INLINE_BYTE_LIMIT` (headroom under the 64 MiB cap). The point is:
never call `message_to_bytes` on a candidate whose `payload_bytes` already exceeds
the ceiling, and treat a framing `ValueError` as "must spool" rather than letting
it abort stripe handling.

---

## Test corrections (Beta: gates 2, 9, 11, 12, 14)

**Gate 2 & Gate 14 — assert the CORRECT reverse-constant shape.**
- Every fixed-skip reverse variant must assert **12 args** = `_constant_prefix` +
  `int32(offset)`, with NO family tail, `best_skips` present, `skip_sequences`
  absent. Remove the current assumption that forward and reverse constant layouts
  are identical.
- Gate 14 (lcg32/minstd full-mode 4-phase): phase lengths become
  `constant=11+tail+1`, `reverse-constant=12`, `forward-hybrid=HYBRID_FWD_LEN`,
  `reverse-hybrid=14`. Fix the `expected_len` map accordingly.

**Gate 9 — actually route two assignments through the executor.**
Current gate calls `ResidueResolver.resolve()` directly. Instead, drive TWO
`stripe_assign`s with DIFFERENT window params through one `SieveExecutor` (or one
daemon) and prove `execute()` requests a fresh residue identity per assignment
(inject a resolver/loader that records the identity key it was asked for; assert
two distinct keys and the correct residues used each time). Keep the direct-unit
checks too, but the gate must exercise the assignment path.

**Gate 11 — approach the REAL 64 MiB cap without lowering INLINE_BYTE_LIMIT.**
Build a payload whose genuine encoded inline message exceeds 64 MiB (e.g. enough
hybrid survivors with real skip-sequences) and assert it spools via the
`ValueError`-catch path — NOT by temporarily shrinking the threshold. The
size-vs-count check can stay, but at least one case must cross the true protocol
cap so the pre-spool-exception bug can't hide.

**Gate 12 — subsequent assignment through the SAME instance.**
Restore the launch hook on the SAME executor/worker (not a fresh `ex2`) and
execute again, proving the daemon that survived the exception serves the next
assignment.

---

## Non-blocking interface clarification (Beta)

`ResidueResolver` currently supports `dataset + window_size + optional
residue_sha256` but no independent residue reference (`residue_path` /
`residue_reference` / inline residues). Resolve ONE of:
- (a) document that Phase 4 ALWAYS supplies dataset/window fields (add an explicit
  note + a guard that fails clearly if it doesn't), OR
- (b) implement an alternate residue-reference path.
Pick (a) unless Phase 4's assignment contract is already known to send a bare
residue reference — (a) is smaller and matches the current coordinator direction.
State which you chose in the changelog. This is non-blocking but must be addressed.

---

## Workflow

Iterate the edit→harness loop on 101 until green — and this time the reverse
variants MUST be exercised: extend the GPU smoke (if a device is present) to launch
at least one reverse-CONSTANT variant (e.g. `java_lcg_reverse`) so a 12-vs-14 arg
error would actually surface on hardware, not just in a CPU shape assertion. GPU
gate stays skippable on CPU-only boxes. **Do NOT commit or push** — Michael commits
after Beta re-approves. Update the changelog with both fixes, the four test
corrections, the reverse-ABI audit note, and the (a)/(b) residue-reference choice.
Report when green and STOP for review.
