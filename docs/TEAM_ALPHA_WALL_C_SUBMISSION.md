# TEAM ALPHA → TEAM BETA — Wall C is already satisfied; recommend removing it as a Phase 6 precondition

**Re:** `docs/KNOWN_ANSWER_VALIDATION_INVENTORY.md` and pre-repository evidence recovered from
Michael's archive. Read-only; nothing changed, nothing executed.

**Alpha scoped Wall C as new work. That was wrong.** Known-answer validation is not a gap to be
filled — it is **documented pre-repository practice**, it was performed across the PRNG registry
during pipeline development, and its method is precisely what Beta's Wall C asks for.

**Alpha recommends Wall C be struck as a Phase 6 precondition**, with one small piece of genuine
follow-on work identified in §4.

---

## 1. The decisive evidence — and why the repository could never have shown it

`prng_registry.py` is present in the repository's **initial commit, `0101306`, 2025-11-29**. The
44-PRNG build and validation therefore **predate the repository entirely.** The oldest surviving
session record is **S73, 2026-02-08**. **The repository's history begins after the work was
finished and cannot evidence it either way.**

The evidence survives in Michael's archive — `instructions.txt`, **2025-12-18**, the operating
document of the period:

```
# Verify the sieve finds it
python3 coordinator.py test_seed42_known.json --method residue_sieve \
    --prng-type xorshift32 --seeds 100000 --offset 0
# Expected: Seed 42 found with 100% match rate
```

And, as **step 1 of the Basic Analysis Workflow** — the routine check performed *before*
production analysis, not a one-off experiment:

```
# 1. Test sieve with known seed (verification)
# Expected: Seed 42 found with 100% match rate in ~47 seconds
```

**Plant a known seed → generate the draws → state the expected result in advance → verify
recovery.** That is textbook known-answer validation. `--prng-type` parameterises it across the
registry; the same workflow was run for the PRNG families as they were brought up. The document
also records the outcome of the period: *"PRODUCTION READY — Forward Sieve Verified on All 26
GPUs."*

**Michael's account is confirmed, and Alpha's earlier qualification is withdrawn.** Alpha
previously wrote that the evidence did not support all 44 being known-answer verified. That
statement was about **repository artifacts**, and it was presented as though it bore on what was
done. It does not. The correct statement is that the repository postdates the work.

## 2. This also explains a finding Alpha misread two days ago

`docs/DAILY3_CONSUMER_CONTRACT_v1.md` flagged `full_state` as a hazard: every sieve loader reads
`entry.get("full_state", entry["draw"])` and **no production record carries the field**, so a
producer emitting it would silently replace the residue stream.

The pre-repo fixture generator shows why it exists:

```python
"full_state": int(state)  # Critical for multi-modulo validation
```

**`full_state` is the known-answer harness's hook.** Test fixtures carried full PRNG state so the
sieve's multi-modulo check could be validated against a known seed. It is not a stray field — it
is deliberate support for exactly the validation Wall C describes, still honoured by every
loader.

The consumer-contract warning stands (a *production* producer must not emit it), but its
characterisation should be corrected from *"unexplained field"* to *"the validation harness's
documented entry point."*

## 3. What the repository does contain, consistent with the above

From the inventory:

- **`pa_sieve_validation_harness.py`** (S143, 352 lines) — its own `java_lcg`, its own CPU
  brute-force bidirectional sieve (*"slow but provably correct"*), three tiers, a planted seed
  that **must** be recovered. **stdlib only** — no coordinator, no registry, no engine.
- **16 independent fixture generators** re-implementing each PRNG inline **with kernel-correct
  skip semantics** (skip between every draw, matching `prng_registry.py:987-989`).
- **Planted-seed tests** across mt19937 / xorshift32, constant **and** hybrid; plus a
  **coordinator-free direct-RawKernel** harness.
- **`fix_cpu_reference.py`** — fixed a real `uint64` overflow in `xoshiro256pp_reverse` and
  **states its expected output verbatim**. Direct evidence the references were exercised against
  known answers and caught a bug.

**Method, Beta's decisive question:** *nothing computed its expected answer by running the
engine and reading it back.* The references are independent.

**Currency:** `miner/range_miner_worker.py:837` imports `sieve_gpu_worker._get_kernel`, which
compiles the registry's `kernel_source`. **The miner runs the same kernel source the original
campaign exercised**, so those results transfer across the RANGE-MINER cutover.

## 4. What is genuinely worth doing — and it is not a wall

One item has no precursor: **nothing has compared a *miner-produced* survivor set against an
independently computed one.** The legacy sieve was validated this way; the miner has not been.
Given §3's currency finding the risk is low, but it is the one honest gap.

**Alpha proposes it as a bounded task, not a certification wall:** plant a seed, run it through
the miner, verify recovery against `pa_sieve_validation_harness.py` — which first needs ~20 lines
of alignment to the production kernel (it uses the Java seed-scramble and `>>17`; production uses
raw seed and `>>16`, and its reverse should be forward iteration over reversed residues rather
than modular-inverse backward stepping). **An afternoon.**

Two smaller items, recorded rather than scheduled: fixtures are gitignored and would need
regenerating in-test, and several legacy harnesses print and exit 0 regardless
(`test_all_hybrids.sh` prints ✅ unconditionally) — VIR-3 non-compliant.

## 5. One artifact that must not be cited

`reverse_kernel_test_results.txt` is **20/20 `BOTH ZERO`** — no clean control, no positive
control, recorded as a test result. **Vacuous under VIR-2.** It is unrelated to the validation
above and must not be offered as correctness evidence. Alpha requests it be **marked superseded
in place** rather than left to be found and trusted.

## 6. Rulings requested

1. **Strike Wall C as a Phase 6 precondition.** Its method is documented pre-repository practice,
   its references are independent, and the artifacts consistent with it are in the tree. Alpha
   does not consider it an open question.
2. **Approve the miner-path known-answer check (§4) as a bounded task**, sequenced with the
   remaining Phase 6 work rather than gating it.
3. **Confirm `reverse_kernel_test_results.txt` may not be cited**, and rule on marking it
   superseded in place.
4. **Correct the `full_state` characterisation** in `DAILY3_CONSUMER_CONTRACT_v1.md` per §2 — the
   warning stands, the explanation was wrong.
5. **Registry count:** the live registry is **44**, not 46. Eight documents are stale and
   `test_ALL_46_prngs_10M.sh` **would hard-fail**. Docs-only; Alpha requests it fold into the
   chapter track.

## 7. Alpha process note

This is the **fifth** absence claim falsified in this working period, and the pattern is now
specific enough to name: **Alpha's searches are repository-scoped, and this project's foundation
predates its repository.** The skill's VIR-6 corollary covers host state and deployed files; it
did not cover *history that begins after the fact*. Alpha will add that surface — pre-repository
archives held by Michael — to the standing scope declaration, and will ask before concluding that
foundational work was never done.

## 8. VIR declaration

The inventory was repository-scoped: working tree plus **full git history across all branches**,
including `git log --all --diff-filter=D` and targeted recovery of `fix_cpu_reference.py` from
`a076602^`. **Nothing was executed** — "would it run today" verdicts are static analysis.
Registry counts and the `java_lcg_cpu` divergence (`skip=3` → cpu `[737,468,925…]` vs kernel
`[737,265,282…]`) come from a live import on VM 101. **The pre-repository evidence in §1–2 comes
from Michael's `~/Downloads/PRNG/instructions.txt`, dated 2025-12-18 — a surface no
repository-scoped audit can reach.** Unavailable: the three rig CT100s, `.127` frozen bare-metal,
host systemd/cron, and any further pre-repository material not held in that archive.
