# PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md — REV2

> **REV2 incorporates all four Team Beta corrections.** If you are reading a copy
> without this banner, it is the superseded REV1 draft — discard it.
>
> | correction | where |
> |---|---|
> | 1. active-path removal evidence | §3, "Active-path removal confirmed" |
> | 2. hybrid-to-zero collapse mechanism | §1.2 |
> | 3. D3.0 did **not** implement the sidecar | §4 |
> | 4. regeneration on the completed D6 canonical path | §4, "Approved sequencing" |
>
> Added in REV2 beyond the corrections: §4a owner testimony, §4b binding prior
> policy, §4c companion coverage-tracker context.


**Purpose:** discharge the Team Beta Ruling F condition for the historical
prior accumulator, unblocking D3.5.
**Disposition: NOT CERTIFIABLE — archived, clean start.** This selects Team
Beta's option 3 (reject and start clean). It supersedes an earlier Team Alpha
draft that proposed *"verified canonical-compatible; no migration required"* —
that draft's premise was disproven by the evidence below and is withdrawn.

---

## 1. Why certification failed

Team Beta's blocking question was whether all 20,949 rows in
`bidirectional_survivors_all.npz` originate from a run of proven identity and
mode. They do not, and the chain cannot be closed. Three independent
disqualifying findings, any one of which is sufficient:

### 1.1 The May 11 run contributed zero rows

The producing-run analysis (artifact bound to
`S174_D1_FT073_50K_425M_20260511_192244` by a 313 ms timing chain and the
launcher's own bundle record) is sound but **irrelevant to the data**. That
run's accumulator summary reads:

```
[S145-R1 v2][NPZ ACCUMULATOR] 20,949 total survivors across all runs
   Prior kept:   20,949
   Net new:      +0
   Superseded:   0
```

Every row was already present. Proving that run was `--prng-type java_lcg`
certifies an empty contribution. All three May 11 runs show the identical
pattern.

### 1.2 Variable mode was enabled across the lineage

`--test-both-modes` is present in the **actual optimizer argv** — not merely in
logged help text — for nearly every run from 2026-03-15 through 2026-05-01,
including `s145r1_smoke_phase1.log`, the origin run that created the first 352
rows. This is not a constant-only lineage.

The final artifact nevertheless contains `skip_mode = {0}` and zero
variable-mode rows. **The all-zero mode/type columns do not merely lack
explanation: the pre-D3.0 writer had a known mechanism capable of collapsing
Java-LCG hybrid survivors into ID 0**, so variable rows may be present but
permanently mislabeled. The copied encoding table lacked `java_lcg_hybrid` and
its `.get(..., 0)` fallback silently encoded unknown identities as `0` (the
defect D3.0 corrected). The artifact therefore cannot distinguish between:

```text
only constant survivors
constant AND variable survivors, both encoded as 0
```

"The hybrid pass presumably found no bidirectional survivors" is a hypothesis,
not evidence, and Ruling F requires evidence.

### 1.3 The file was repeatedly overwritten, not accumulated

Exactly **one** growth event exists in the entire log history
(`Net new: +352`, 2026-03-15). The row count nonetheless moved
352 → 666 → 674 → 20,912 → 20,914 → 20,916 → 20,949.

Those changes were whole-file replacements. Dozens of runs record:

```
⚠️  [S145-R1 v2][NPZ ACCUMULATOR] Failed: index N is out of bounds for axis 0 with size 0
Falling back to per-run convert_survivors_to_binary.py
```

The fallback **overwrites** the artifact with a single run's survivors rather
than merging. The current 20,949 rows first appear on 2026-05-01, written by a
fallback overwrite from a run whose argv included `--test-both-modes`.

Compounding this: the fallback writer is `convert_survivors_to_binary.py`, the
same writer carrying the **cross-direction match-rate fallback** identified
during D3 (a missing `forward_match_rate` silently populated from
`reverse_match_rate`). It cannot be shown that path did not fire.

`size 0` in those failures is the pre-D3.0 empty-NPZ defect: the one-array
`seeds=[]` artifact being indexed as though rectangular.

**Conclusion:** every candidate ancestor is itself an overwrite produced by a
both-modes run through a writer with a known fabrication path. There is no
clean origin to recurse to.

## 2. Collateral finding — the accumulator never accumulated

The artifact's design purpose was to carry survivors across the multi-part full
seed sweep. The log record shows that purpose was **never actually served**:
one real accumulation event in five months, with every other count change being
a reset. The `index N out of bounds` failures begin 2026-03-28 and recur
through 2026-05-01.

**Consequence for D3.5:** D3.5 is not porting a working cross-run accumulator.
It is **implementing cross-run accumulation correctly for the first time**. The
D3.5 brief should state this explicitly so the L3 prior-merge path is treated
as new construction requiring full gate coverage, not as a behavior-preserving
port. The pre-D3.0 empty-NPZ defect that triggered the failure cascade is
already fixed (rectangular 22-array empty output), which removes the specific
trigger — but the merge path itself has no proven-good production history to
regress against.

## 3. Archive

Archived, not deleted, on VM101 at
`/home/michael/distributed_prng_analysis/archive/pre_d3_accumulator_20260725/`
(timestamps preserved via `cp -p`):

```text
f6dc651f8a794c0d5b918432b437e6c0b37786f1a973e1e7d622b12939704233  bidirectional_survivors_all.npz
f6dc651f8a794c0d5b918432b437e6c0b37786f1a973e1e7d622b12939704233  bidirectional_survivors_binary.npz
796b339d97cfe8383e7527bdaee707237f1910304165f58f1964d074d4ae0488  bidirectional_survivors_all.npz.ckpt.tmp.npz
dbf8d5d760b9f04499ab65544cf4da9dd14d6775fe3f89185d4826f22e66cd83  bidirectional_survivors_all.npz.flush.tmp.npz
```

The two canonical artifacts are byte-identical to each other. The two `.tmp`
files carry distinct hashes and are stale partials (2026-03-22 and 2026-05-01)
lacking a `prng_type` column; they are preserved as separate forensic objects.

**Active-path removal confirmed (Team Beta blocking condition).** After the
archive hashes were verified, the originals were removed from the active
working tree:

```text
active canonical prior            : absent
  bidirectional_survivors_all.npz      -> absent
  bidirectional_survivors_binary.npz   -> absent
active stale temporary artifacts  : absent
  bidirectional_survivors_all.npz.ckpt.tmp.npz   -> absent
  bidirectional_survivors_all.npz.flush.tmp.npz  -> absent
archive hashes                    : verified identical BEFORE and AFTER removal
```

Runtime state can therefore no longer load the rejected artifact. A root scan
found only four unrelated January test fixtures (`test_output.npz`,
`test_survivors_{500,5000,20000}.npz`); none carries a canonical accumulator
filename, so no production path can consume them. They are noted for the S110
root-cleanup backlog, not for this disposition.

Artifact facts of record: 22 keys, 20,949 rows, `prng_type = {0}`,
`skip_mode = {0}`, 94,134 bytes, mtime `2026-05-11 19:24:23`.

## 4. Disposition and operational consequence

```text
Ruling F option selected : 3 — reject and start clean
Certified prior          : NONE
D3.5 migration path      : NOT REQUIRED — must not be implemented
D3.5 prior-merge default : fail closed on any uncertified prior artifact
```

Going forward, **D3.5 must write the canonical artifact and its provenance
sidecar as one logical operation**, recording the encoding contract, repository
commit, artifact hash, row count and schema version. D3.0 corrected the encoding
seams and made the empty artifact rectangular; it did **not** implement the
provenance sidecar, and no post-D3.0 file should be assumed to carry sidecar
certification. D3.25 fixed candidate ingress; D3.0-B will close the
cross-direction fallback in the legacy writers.

**These 20,949 rows are regenerable Step-1 output, not irreplaceable
measurement.** Regeneration costs one Step-1 run on corrected code. Certifying
them would require further archaeology with a likely-negative result, and even
success would certify data written by a writer with a known fabrication path.

**Approved sequencing (Team Beta):** a newly generated artifact should be
produced by the **completed canonical path**, not by the legacy writer
immediately before that path replaces it.

```text
1. Archive and remove the uncertified prior.        (this document)
2. Implement D3.5 with no prior present.
3. Land D4 and D5.
4. Land D6 production wiring.
5. Land D3.0-B before Phase 6 certification.
6. Run the Zeus smoke trial on the completed canonical path.
7. Treat that run as the first certified accumulator baseline.
```

D3.5 is tested against **synthetic** certified priors and malformed/uncertified
priors; it does not require a regenerated real prior in order to be implemented.
The archived copy remains available for forensic comparison only and carries no
certification.

## 4a. Owner testimony — corroborating cause

The project owner confirms that the entire March-May window was a **debugging
phase**: the PWC transport work (SSH → TCP) and the original distribution method
were failing repeatedly, and the runs of that period were crash-reproduction and
transport-diagnosis attempts rather than production sweeps. This is the direct
cause of the abnormal accumulator record — the fallback-overwrite cascade
documented in §1.3 is the signature of that campaign, not of normal operation.
Owner position: not something to build on.

This is recorded as testimony corroborating the log evidence, not as a
substitute for it.

## 4b. Binding D3.5 prior policy (Team Beta)

```text
No prior file                                  -> start with an empty global accumulator
Prior file with valid matching sidecar         -> validate and perform L3 merge
Prior file without sidecar                     -> fail closed
Prior file with hash/schema/encoding mismatch  -> fail closed
Historical pre-D3 accumulator                  -> never migrate, reinterpret, or silently import
```

**No filename-based trust. No compatibility exception for the archived
20,949-row artifact.** It may be used only for forensic comparison, never as
candidate input.

## 4c. Companion-artifact context (non-blocking, for D3.5 design)

Recorded because it bears on the D3.5 merge contract, not as further provenance
investigation. The accumulator's companion is the seed **coverage tracker** —
`prng_analysis.db`, table `exhaustive_progress` — which records which seed ranges
were swept. Its state at `70cd6f0`:

- 15 rows for `java_lcg`/`bidirectional`, spanning `0 -> 16,106,127,360`
  (**3.75x the 2^32 target** of the S145-R1 Progressive Empirical Sweep; the
  boundary was crossed around 2026-04-24 and advancing continued to 2026-05-03);
- a **coverage gap**: row 1 ends at `425,000,000`, row 2 begins at
  `1,073,741,824` — the intervening ~648M is unclaimed, consistent with
  `reset_coverage_s152.py` deleting rows `>= 660,000,000` on 2026-03-22;
- quality columns never populated: `best_score = 0.0` on 14 of 15 rows,
  `best_seed = NULL` throughout;
- row 1 (`0 -> 425,000,000`) has `last_updated = 2026-05-11T19:24:23.706519` —
  **16 ms after the NPZ write** — and its bounds match the D1 run's
  `--seed-start 0 --max-seeds 425000000`. The crash-reproduction campaign was
  overwriting the first coverage row on each attempt.

**Implication for D3.5:** survivors and coverage are a consistency pair —
survivors should exist only for swept ranges — and neither the old writer nor
the tracker enforced that relationship. Since D3.5 builds the merge path fresh,
the brief should state whether that invariant is in or out of scope. The
tracker's own integrity (the gap, the overshoot, the unpopulated columns) is a
separate operational question that should be settled before Phase 6 results are
interpreted against "already covered" claims.

## 5. Record

```text
repository HEAD        70cd6f078709a601832c5609da8d066c68cc87a0  (D3.25)
inspection timestamp   2026-07-25T16:55:12-07:00
archived               2026-07-25
evidence sources       logs/s145r1_smoke_phase1.log            (origin, +352, both-modes in argv)
                       logs/s172_pool8_100k_clean_t5_1543.log  (first 20,949, both-modes in argv)
                       logs/S174_D1_FT073_50K_425M_20260511_192244{,_launcher}.log, _summary.json
                       full accumulator-block sweep across all logs/*.log, oldest first
```

— Team Alpha (Claude), 2026-07-25
