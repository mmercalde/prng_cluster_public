---
Status: Superseded
Phase: S172 Phase 3
Applies to: range_miner_worker.py
Superseded by: PHASE3_FIX_BRIEF_REV2.md
---

# S172 Phase 3 — Initial Review (rev-1 REJECTION)

Team Beta rejected the first Phase 3 implementation with five release-blocking
findings. Directional architecture was sound (declarative kernel-arg builders,
uint64 Java ABI preservation, uncovered-family hard-fail before cupy/launch,
sub-stripe partitioning, serialized socket writes + heartbeat thread, coordinator
retry kept out of the worker — all approved). The five blockers:

## 1. Stale residue window (correctness)
`SieveExecutor` loaded `draws` once at process start and reused `self.draws` for
every later `stripe_assign`. A long-lived worker serves different Window
Optimizer trials where `window_size`/`sessions`/`offset` — and thus residues —
change. This silently runs a valid kernel against WRONG-TRIAL data (worse than a
crash). Required: per-assignment residue resolution keyed by an immutable residue
reference/hash.

## 2. Large-result handling destroyed survivors (data loss)
For `count > INLINE_SURVIVOR_LIMIT` the code set `msg.inline = None` and sent
neither survivors nor a spool path — coordinator got a count only, data gone.
Second defect: the decision was on survivor COUNT, not serialized byte size, so a
hybrid survivor (full skip-sequence) could exceed the 64 MiB frame under 50k
survivors. Required: real atomic spool with path/size/sha256, size-based
selection under `MAX_FRAME_BYTES`.

## 3. GPU cleanup incomplete + not exception-safe (stability)
rev-1 did only `del` + `gc.collect()`, only on the success path. The proven
worker also does torch sync/cache-clear and CuPy default + pinned pool releases,
after explicit array deletion. A long-lived daemon that skips cleanup on the
exception path accumulates VRAM/VM pressure — the S154 OOM failure mode. Required:
`try/finally` with a shared best-effort cleanup after every sub-stripe.

## 4. Advertised family support contradicted `test_both_modes` (spec)
The worker advertised six covered base families but `_reject_hybrid()` refused
hybrid variants for all but java_lcg. Frozen acceptance §11.I requires ≥3 base
PRNGs (java_lcg, lcg32, minstd) to pass with `test_both_modes=True`, and
`resolve_kernel_families()` auto-adds `_hybrid`/`_hybrid_reverse`. Implementation
and tests could not satisfy acceptance simultaneously.

**Ruling — Route B (binding):** implement the non-Java hybrid builders. Route A
(narrow the advertised set) would weaken a frozen acceptance requirement and
would require a formal erratum — not an implementation choice. Beta acknowledged
a spec defect: §5.3 gave only the Java hybrid ABI, not the non-Java signatures,
which is what let `_reject_hybrid()` look consistent. Audit each hybrid kernel
ABI from the registry rather than extrapolating Java's layout. Minimum set:
java_lcg, lcg32, minstd (§11.I). Handshake must advertise exact variants.

## 5. Harness missed the dangerous paths
No tests for: two different draw windows, a spooled result, a near-64-MiB hybrid
inline, cleanup after a GPU exception, exact capability advertisement, or a real
non-Java `test_both_modes` run. A skippable GPU gate means "green" can occur on a
CPU-only box — contract validation, not deploy readiness.

## Ruling
Reject rev-1. Approve after: per-assignment residue correctness; real spool with
size/hash; exception-safe full cleanup; Route B hybrid support; blocking tests
for each.
