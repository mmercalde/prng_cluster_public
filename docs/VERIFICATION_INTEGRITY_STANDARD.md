# VERIFICATION_INTEGRITY_STANDARD.md — VIR-1 … VIR-5

**Status:** standing project rule. Adopted by Team Beta in the Phase 6.0 final ruling
(2026-07-30) from a Team Alpha proposal. Referenced by every implementation brief.

**Scope:** all gates, harnesses, mutation experiments, smoke tests, health checks, and
acceptance evidence in this project — Team Alpha, Team Beta, and any implementer.

---

## Why this exists

Three incidents in three consecutive deliverables shared one shape: **a check that was
not checking, presenting as a check that passed.**

| # | deliverable | the non-check | how it presented |
|---|---|---|---|
| 1 | D6.1 | the mutant harness passed a mutated module as an *argument* while the gates built their own from production | **M2 "survived"** — the mutation experiment proved nothing |
| 2 | D6.1 | `_flush_npz_incremental` failed on **every** run for months | a **non-fatal warning** nobody treated as a failure |
| 3 | Phase 6.0 | cleanup `pkill -f` matched its own shell and killed it mid-verification | **silence** — indistinguishable from a quiet pass |

Each was caught by accident: a suspicious survivor, a log line read closely, a
truncated report. The principles below already existed in practice — D6.1's sentinel
audit, the four-part mutant rule, Phase 6.0's refusal to run a container-side `dmesg`
that would return empty regardless of GPU state. **This document makes them
contractual rather than dependent on a reviewer remembering.**

---

## VIR-1 — Verification must prove its own execution

> **Every verification must provide evidence that it started, exercised its intended
> subject, and completed. Silence, truncated output, reporter termination, an
> inaccessible observation surface, or the absence of a completion record may never be
> interpreted as a pass.**

## VIR-2 — Potentially vacuous detectors require controls

When a detector could pass *without observing the relevant path*, its acceptance proof
must include all four:

1. **Execution proof** — the intended production object or path was actually reached.
2. **Clean control** — the unmodified valid case passes.
3. **Fault-injection (positive) control** — a deliberately injected relevant defect
   *is* detected.
4. **Detector independence** — the injected defect is not being caught accidentally by
   an unrelated failure *before* the intended detector runs.

**Terminology (Beta correction, use precisely):**
- **Positive / fault-injection control** — proves the detector **fires** when a defect
  is present.
- **Clean / negative control** — proves the detector does **not** fire when no defect
  exists.

Both are required. They are not interchangeable, and "negative control" must not be
used for fault injection.

This formalizes the existing four-part mutant rule:

```
applies-once · mutated-path · detector-clean · injected-defect
```

**A mutant that changes the wrong module, an argument production code never consumes,
or an unreachable branch is not a surviving mutant — it is an invalid mutation
experiment.** Report it as invalid, not as a survivor.

## VIR-3 — No silence-as-pass

Every harness must terminate in an explicit, machine-detectable state:

```
PASS | FAIL | UNAVAILABLE | INCOMPLETE
```

**Only `PASS` satisfies an acceptance item.**

- `UNAVAILABLE` — the observation surface could not be accessed.
- `INCOMPLETE` — timeout, reporter death, truncated output, missing completion marker,
  or cleanup terminating the supervising shell.

Neither may be silently converted to `PASS`.

For critical harnesses, emit a final sentinel containing at least:

```
verification_id · start marker · completion marker ·
exit status · executed-path confirmation · result
```

**A missing final sentinel is failure or incomplete verification, never success.**

## VIR-4 — Cleanup must not be able to kill its reporter

Broad pattern-matched cleanup such as `pkill -f <pattern>` must **not** be issued from
a shell whose own command line can match that pattern.

Prefer, in order: captured child PIDs · process handles · pidfiles · exact executable
and argument matching · a supervising process outside the target process group.

Cleanup failure remains **secondary** to the primary test result — but cleanup must
never terminate or silence the reporting authority.

## VIR-5 — Unobservable is not clean

A check based on logs, counters, hardware telemetry, privileges, or namespace
visibility must **first prove the observation source is available.**

```
empty host kernel log after an authoritative query   → potentially clean
empty container kernel log, host logs inaccessible   → UNAVAILABLE
zero RAS counters returned by supported hardware     → potentially clean
RAS interface absent on unsupported hardware         → UNAVAILABLE
```

**Reference example (Beta-designated):** Phase 6.0's handling of the unprivileged-LXC
kernel-log restriction. `dmesg` was denied, `/dev/kmsg` absent, `journalctl -k` empty —
inside the container those return empty *regardless of what the GPU did*. Running it
anyway would have produced a green line that meant nothing. It was reported
`UNAVAILABLE` with the exact command for an authoritative run, and the failure class
was evidenced through surfaces that *do* move on a fault. The gap was then closed from
the Proxmox host, where the `amdgpu` driver actually lives.

Equally: RAS counters were reported **unavailable** on consumer RDNA2 rather than
"clean", so their absence could not be misread as a passing check.

---

## Required block in every implementation brief

Every future brief carries this section; every return package answers it:

```
Verification-integrity controls (VIR-1…5):
- execution proof:
- clean control:
- fault-injection control:
- completion sentinel:
- unavailable-observer behavior:
```

## Reviewer checklist

- [ ] Did every acceptance item terminate in an explicit `PASS`?
- [ ] Could any passing check have passed **without** observing its subject?
- [ ] Does each mutation experiment prove the mutated path executed in the object
      production actually uses?
- [ ] Is any "clean" result actually `UNAVAILABLE`?
- [ ] Can cleanup kill the reporter?
- [ ] Is a missing/truncated report being read as success?

---

**Precedents already in the codebase** (these work — extend the pattern, don't reinvent
it): D6.1's sentinel audit used an empty sentinel directory as a control and caught a
genuine pre-isolation leak; D6.1's `G-COMMENT-TRUTH` fails in **both** directions (a
retracted claim reappearing *or* a disclaimer going missing); Phase 6.0's
`G-TRANSACTION-IDENTITY` asserts both that seed-set comparison *sees* agreement (so the
gate has teeth) *and* that the detector reports the interruption.
