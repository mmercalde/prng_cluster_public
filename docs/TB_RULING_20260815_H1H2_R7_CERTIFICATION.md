# Team Beta ruling — **R7 H1/H2 INSTRUMENTATION CERTIFIED**

**PASS. The R1→R7 technical review is closed.**

I independently verified the exact R7 artifact:

```text
test_s172_h1h2_instrumentation_R7_EXACT.py
bytes       173,172
sha256      a2b69fc64c11175d1d1b6a599cde34a9dc2608010cb1da1d760e6fc82bc835b2
_ast_calls_to occurrences   8
check(...) CALL SITES       62
def check(...)              1
Python compile              PASS
```

Those values exactly match Alpha's submission.

### Independent executable closure

I extracted the actual R7 helper implementations from the submitted artifact and ran the
counterexamples directly.

**R7-1 / E4 — PASS.** The nested-def counterexample that defeated R6 now reds:

```text
def enqueue_staging(self):
    with self._admission_lock:
        def later():
            self._defer_locked(entry)     # defined, never executed
        self.note_stripe_frame_deferred(...)
```

Result:

```text
enqueue_staging makes no `_defer_locked(...)` CALL —
there is no insertion, so no critical section can be identified
```

The legitimate same-lock implementation passes. A call inside a comprehension is counted, as
intended. An actually invoked nested definition conservatively reds, matching Alpha's explicitly
declared fail-safe boundary.

**R7-2 / N1 — PASS.** I executed all important reach shapes:

```text
direct fail_trial                     RED
getattr(self, 'fail_trial')(...)      RED
stored self.fail_trial reference      RED
computed indirect callee              RED
getattr(self, dynamic_name)           RED
me=self; getattr(me, dynamic_name)    RED

getattr(msg, 'stripe_id', None)       CLEAN
getattr(msg, dynamic_name, None)      CLEAN
```

The exact R6 counterexample and the additional self-alias counterexample are both closed without
reverting to the coarse "all getattr is suspicious" mistake.

**R7-3 / N6 — PASS.** Direct execution produced offenders for all of these:

```text
except: return []
except: result=[]; return result
result=[]; alias=result; return alias
result: list=[]; return result
result=[]; box={'v': result}; return box['v']
except: return 0
except: return {'status':'UNAVAILABLE', 'active_count':0}
```

The proper:

```python
{'status': 'UNAVAILABLE',
 'active_count': None,
 'stripes': None,
 'error': ...}
```

remains accepted.

That independently confirms both Beta's R6 findings **and** the additional container-taint defect
Alpha found while attacking its own R7 repair.

**R7-4 / claim correction — PASS.** `_ast_calls_to` now claims only E4 and N2 coverage and
explicitly excludes `_single_call_offenders` and `_renewal_clock_offenders`. The previous overclaim
is gone.

### N2 inheritance also confirmed

The same-scope repair propagates correctly into `_renewal_site_offenders`:

```text
comment masquerading as second renewal    RED
third actual renewal                      RED
missing source discriminator               RED
nested-def renewal                         ignored correctly
two legitimate live-shape renewals         CLEAN
```

So R7-1 did not repair E4 while accidentally corrupting N2.

### Declared boundaries

I accept the four boundaries in the R7 report as **properly bounded claims, not hidden proof
holes**: receiver identity is not established by `_ast_calls_to`; genuinely invoked nested
definitions can false-red; N1 dynamic lookup is scoped to `self` and its aliases; and N6 is not
claiming general interprocedural or mutation-aware taint analysis.

Those limits matter because the review has now reached the point where distinguishing **an actual
contradiction of the asserted invariant** from **arbitrary incompleteness of a static analyzer** is
essential. R7 does that correctly.

I am therefore **not opening another round for hypothetical AST constructions outside the explicitly
bounded proof model**.

### Regression and production evidence

Alpha reports the production miner digests remain:

```text
worker       043522e96b44855f04540b1d2bdb5db003f3428785d7c98e9bfc073ff5a8100d
coordinator  b97ce5f9b2dc455b615f130d9575c1a285fd43e66e7ed230739849a83d35ab67
```

and supplies the full regression matrix, including H1/H2 **62/62**, with only the already-understood
Phase-4 Gate 22 development-tree red.

I did not independently execute those production-dependent suites here because the corresponding
live miner tree is not part of this artifact environment. I **accept Alpha's supplied
regression/digest evidence as the execution evidence for those portions**; the structural R7 closure
itself was independently exercised against the submitted artifact.

## Final disposition

```text
R7 ARTIFACT IDENTITY                     PASS
62 check(...) call sites                 PASS
R7-1 same-scope execution proof          PASS
R7-2 dynamic/self-alias reach proof      PASS
R7-3 N6 fixed-point taint proof          PASS
R7-4 claim correction                    PASS
R7 self-sweep counterexamples            PASS
declared detector boundaries             ACCEPTED

PRODUCTION DEFECT FOUND                  NO
PRODUCTION CHANGE REQUIRED               NO

H1/H2 INSTRUMENTATION R1–R7              CERTIFIED

R8 REQUIRED                              NO

ARCHIVE FLEET                            AUTHORIZED / REQUIRED NEXT
COMMIT + DUAL PUSH                       AUTHORIZED AFTER ARCHIVE
CLEAN-TREE PROOF                         REQUIRED AFTER COMMIT
DEPLOY ALL TEN GOVERNED FILES            AFTER CLEAN-TREE PROOF
PARTIAL COORDINATOR/WORKER DEPLOYMENT    FORBIDDEN
PARITY                                   REQUIRE 30 MATCH
                                         0 MISMATCH
                                         0 UNAVAILABLE
NORMAL PRELAUNCH AUTHORITY               REQUIRED AFTER PARITY
ATTEMPT 7                                HELD UNTIL PRELAUNCH AUTHORITY
```

**Team Beta certification statement:**

> **S172 H1/H2 discrimination instrumentation R7 is CERTIFIED.** The known verification defects
> identified through R1–R7 are closed within the expressly bounded proof model. No further
> test-authority amendment is required before proceeding through the governed archive →
> commit/push → clean-tree → full deployment → parity → prelaunch sequence. Attempt 7 remains held
> until that sequence completes.

This one is done.
