# Team Beta Review — WATCHER Retrain-KPI Governance Findings, S176

## Verification basis

Team Beta independently checked the central S176 claims against the current live GitHub `main` branch of:

```text
mmercalde/prng_cluster_public
```

Verified repository head:

```text
0c3166a630be321809f415bb28af28e319d0fe1b
```

The KPI findings below are therefore based on the current repository code, not solely on documentation, memory, or Team Alpha's report.

One limitation remains:

> GitHub `main` may not include uncommitted changes currently present on Zeus.

Where the current live repository disagrees with the S176 findings, that disagreement is identified explicitly below.

---

# 1. Executive ruling

**Team Beta approves the central structural findings of S176.**

The current WATCHER Layer-2 retrain governance is not calibrated to real TFM operation, and TFM presently has **no empirical operating baseline** from which WATCHER can determine what "healthy," "degraded," or "collapsed" performance means.

The correct initial architecture is therefore:

```text
BOOTSTRAP → CALIBRATING → GOVERNED
```

TFM should not begin with arbitrary synthetic-era performance thresholds, but it also should not begin without governance.

During bootstrap:

* mathematical null baselines are available;
* structural and pipeline-integrity gates can be active;
* prospective performance KPIs must be collected;
* uncalibrated retrain triggers should operate in audit-only mode;
* a causally correct historical walk-forward should establish the first empirical TFM baseline.

No final numerical replacement thresholds should be activated until that baseline exists.

---

# 2. Findings independently confirmed from live repository code

## 2.1 `current_hit_rate` is instantaneous

The live diagnostic computes:

```python
current_hit_rate = exact_hits / max(pool_size, 1)
```

For a distinct 20-number prediction pool:

```text
hit draw  → exact_hits = 1 → current_hit_rate = 0.05
miss draw → exact_hits = 0 → current_hit_rate = 0.00
```

The quantity is not a rolling rate over multiple draws. It is a per-draw hit/miss result divided by pool size.

---

## 2.2 `hit_rate_collapse` is an instantaneous miss detector

The live trigger reads the current diagnostic snapshot:

```python
hit_rate = pipeline_health.get("current_hit_rate", 1.0)
collapse_threshold = triggers.get("hit_rate_collapse_threshold", 0.01)

if hit_rate < collapse_threshold:
    ...
```

At pool size 20:

```text
hit  → 0.05 < 0.01 → false
miss → 0.00 < 0.01 → true
```

Therefore, the trigger answers:

> "Did the latest pool miss?"

It does not answer:

> "Has the historical hit rate collapsed?"

This conclusion is deterministic and does not depend on TFM's eventual real predictive performance.

---

## 2.3 `hit_rate_collapse_window=20` is configured but unused by the trigger

The policy contains:

```json
"hit_rate_collapse_threshold": 0.01,
"hit_rate_collapse_window": 20
```

However, the trigger implementation shown above reads only the single current `current_hit_rate` value. It does not read or apply `hit_rate_collapse_window`.

The configured window therefore has no effect on the live collapse trigger.

---

## 2.4 `max_consecutive_misses=5` is actively enforced

The diagnostic increments `consecutive_misses` whenever `exact_hits == 0`:

```python
if exact_hits == 0:
    consecutive_misses = prev_misses + 1
else:
    consecutive_misses = 0
```

The trigger fires when:

```python
consecutive_misses >= max_consecutive_misses
```

with the policy default of five.

Under the uniform Top-20 random null:

p_hit = 20/1000 = 0.02
p_miss = 0.98

The expected wait until the first run of five misses is approximately:

E[T_5] = (1 - q^5) / (p * q^5) ≈ 5.31 draws

Therefore, five consecutive misses is normal under the random null.

Alpha also showed that the trigger remains unsuitable throughout the tested assumed hit-rate range of 2% to 15%.

However, because TFM has not yet established a measured healthy real-data hit rate, the exact wording should be:

> **The five-miss trigger is analytically unsuitable across the tested plausible range and must not be treated as an empirically calibrated TFM degradation threshold.**

It should not be described as empirically proven to false-fire under healthy TFM operation, because "healthy TFM operation" has not yet been measured.

---

## 2.5 `retrain_after_n_draws=10` is implemented as a periodic counter

The policy states:

```json
"retrain_after_n_draws": 10
```

and describes it as:

```text
Minimum draws for statistical significance before retraining
```

The runtime implementation simply checks:

```python
draws_since_retrain >= n_draws_threshold
```

This means the code treats ten draws as a periodic retraining cadence, not as a statistical test.

If the intent is genuinely statistical significance, ten observations are insufficient for estimating low hit rates with useful precision.

If the intent is merely scheduled retraining every ten draws, it should be renamed and documented as a cadence policy rather than a significance threshold.

---

## 2.6 `minimum_hit_rate=0.05` is a synthetic-data target

The current live policy places `minimum_hit_rate` under:

```text
convergence_targets
```

and explicitly describes it as:

```text
System should achieve >5% hit rate on synthetic data
```

Therefore:

* it is not an empirical real-TFM baseline;
* it should not be promoted directly into live real-data governance;
* it cannot currently define healthy TFM operation.

Team Alpha reports that it has no runtime consumer. Team Beta confirmed that the live retrain-trigger and diagnostic paths reviewed do not use it.

Before removing it, Alpha should complete one final repo-wide consumer search on the exact Zeus working tree. Assuming no additional consumer exists, it should be deprecated and replaced by explicit pool-specific baseline fields.

---

## 2.7 Current `pool_coverage` is outcome-space breadth

The live diagnostic computes:

```python
unique_predictions = len(set(...))
pool_coverage = unique_predictions / 1000
```

This measures:

> the fraction of the 000–999 outcome space occupied by unique predictions.

It does not measure:

* historical Hit@K;
* recall;
* probability calibration;
* model weight mass captured by a pool;
* future draw coverage.

The current flag:

```python
if prediction_validation.get("pool_coverage", 0) < 0.01:
    flags.append("LOW_POOL_COVERAGE")
```

is therefore only a low-breadth or degeneracy check.

It should not be interpreted as governance of TFM's prospective predictive performance.

---

# 3. Critical real-data limitation

TFM has not yet completed the required prospective real-data operating cycle under the current system.

Therefore, TFM currently has no empirical distributions for:

```text
Hit@20
Hit@100
Hit@300
miss-run length
survivor churn
confidence drift
window decay
pool stability
weight entropy
calibration correlation
regime-change frequency
```

This does not invalidate the structural S176 findings.

It does mean that S176 cannot legitimately select final replacement thresholds.

Statements such as:

```text
"The trigger fires on 98% of healthy TFM draws"
```

must not be used.

The correct statement is:

```text
"At the uniform random Top-20 null rate of 20/1000 = 0.02,
the instantaneous collapse trigger would fire on approximately
98% of draws."
```

The uniform null is a mathematical baseline. It is not yet the measured healthy TFM baseline.

---

# 4. The baseline problem

When TFM begins real operation, WATCHER has no empirical history from which to judge degradation.

This creates an unavoidable bootstrap problem:

```text
WATCHER needs baselines to govern TFM,
but TFM must run before empirical baselines can exist.
```

The solution is not to invent thresholds.

The solution is to distinguish three types of baselines.

---

## 4.1 Mathematical null baselines

These are available before the first TFM run.

For a 1,000-outcome space:

```text
Random Hit@20  = 20 / 1000  = 0.02
Random Hit@100 = 100 / 1000 = 0.10
Random Hit@300 = 300 / 1000 = 0.30
```

These permit calculation of:

```text
lift@K = observed Hit@K / random Hit@K
```

They answer whether performance is:

* below random;
* approximately random;
* above random.

They do not define healthy TFM performance.

---

## 4.2 Structural and invariant baselines

These can be enforced from the first cycle because they test correctness rather than learned predictive performance.

Examples include:

```text
required files exist
JSON and NPZ artifacts are valid
feature-schema hashes match
model sidecar exists
prediction values are finite
pool sizes match their contracts
duplicate counts remain within policy
weights are normalized
no malformed or empty prediction pool
minimum survivor population exists
stage output is newer than its inputs
```

The live WATCHER already performs several file and content validity checks, including manifest-driven input/output validation.

These gates can remain active during bootstrap.

---

## 4.3 Empirical TFM baselines

These require a historical walk-forward or prospective observations.

They include:

```text
healthy Hit@20/100/300 distributions
healthy variance
normal miss-run distributions
normal survivor churn
normal confidence drift
normal window decay
normal pool stability
normal entropy ranges
normal calibration behavior
```

These values must initially be represented as unknown rather than populated with arbitrary constants.

---

# 5. Required WATCHER governance states

## 5.1 `BOOTSTRAP`

Used when insufficient historical TFM observations exist.

### Active enforcement

```text
pipeline integrity
artifact existence
schema validation
pool non-degeneracy
model loading validity
invalid numeric values
catastrophic execution failures
hard safety constraints
```

### Audit-only monitoring

```text
Hit@20 collapse
Hit@100 collapse
Hit@300 collapse
consecutive misses
survivor churn
confidence drift
window decay
regime shift
LLM confidence
```

During bootstrap, performance KPIs are recorded, but they do not autonomously dispatch retraining based on unvalidated thresholds.

---

## 5.2 `CALIBRATING`

Entered once enough historical walk-forward or prospective observations exist to estimate distributions.

During this state:

* candidate thresholds operate in shadow mode;
* hypothetical trigger actions are recorded;
* false alarms are counted;
* trigger overlap is measured;
* recovery without intervention is measured;
* human review remains required;
* later outcomes are compared against earlier trigger decisions.

---

## 5.3 `GOVERNED`

Entered only after candidate policies demonstrate acceptable behavior.

Required evidence should include:

```text
adequate observation count
stable baseline across multiple folds
acceptable false-trigger frequency
reasonable average time between false alarms
tested cooldown
tested hysteresis
clear mapping from trigger to action
auditable decision history
safe rollback path
```

Only then should performance-based retraining or reruns become autonomous.

---

# 6. Team Beta rulings on the four requested questions

## Q1 — Add governance for TFM's success criteria?

### Ruling: YES, but initially audit-only

WATCHER should collect two distinct classes of metrics.

### Prospective outcome metrics

```text
Hit@20
Hit@100
Hit@300
Lift@20
Lift@100
Lift@300
```

These measure whether future actual draws land in the prediction pools.

### Pool-structure metrics

```text
Top-20 weight share
Top-100 weight share
Top-300 weight share
unique prediction count
outcome-space breadth
prediction entropy
effective pool size
duplicate count
pool-to-pool stability
```

These measure the structure and concentration of predictions.

The two metric classes must remain separate.

A pool may contain high model-weight concentration but still miss future draws. Conversely, a broad pool may achieve high Hit@300 while providing weak Hit@20 concentration.

No fixed retrain threshold should be activated until the empirical baseline phase is complete.

---

## Q2 — Retune or repoint the decorative triggers?

### Ruling: REPOINT architecturally; calibrate empirically

### `hit_rate_collapse`

Repoint from instantaneous:

```text
exact_hits / pool_size
```

to a genuine rolling or sequential Hit@K process.

Do not yet select the final window or collapse threshold.

A nominal 20-draw window is not automatically valid.

Under the random Top-20 null:

P(zero hits in 20 draws) = 0.98^20 ≈ 66.8%

Therefore, even a correctly implemented 20-draw window can appear empty very frequently at low hit rates.

Candidate statistical methods include:

```text
rolling binomial lower-control limit
Wilson lower confidence bound
beta-binomial posterior
CUSUM
SPRT
```

The method and parameters must be selected after walk-forward data exists.

### `max_consecutive_misses`

Do not simply duplicate the rolling hit-rate trigger.

If retained, it should remain a distinct drought/anomaly detector.

Its final run length should be chosen from:

* the empirical miss-run distribution;
* the desired false-alarm horizon;
* acceptable average run length;
* pool size and observed Hit@K.

During bootstrap, it should be disabled for autonomous action or operate audit-only.

---

## Q3 — Wire or remove `minimum_hit_rate`?

### Ruling: DO NOT wire the current `0.05`

The current value is explicitly a synthetic-data convergence target and is not a measured real-TFM floor.

Replace it with explicit pool-specific, baseline-aware configuration.

Example direction:

```json
{
  "kpi_governance": {
    "state": "BOOTSTRAP",
    "hit20": {
      "null_rate": 0.02,
      "empirical_baseline": null,
      "minimum_samples": null,
      "collapse_threshold": null,
      "enforcement": "audit_only"
    },
    "hit100": {
      "null_rate": 0.10,
      "empirical_baseline": null,
      "minimum_samples": null,
      "collapse_threshold": null,
      "enforcement": "audit_only"
    },
    "hit300": {
      "null_rate": 0.30,
      "empirical_baseline": null,
      "minimum_samples": null,
      "collapse_threshold": null,
      "enforcement": "audit_only"
    }
  }
}
```

Once the new schema exists and a final Zeus repo-wide search confirms no additional consumer, remove or deprecate the old generic `minimum_hit_rate`.

---

## Q4 — D1 and D2 config-path defects?

### Ruling: NOT CONFIRMED AS CURRENT DEFECTS ON GITHUB `main`

The S176 report identifies:

```text
D1 — Step 4 output/gate mismatch
D2 — Step 3 output filename mismatch
```

However, current live `main` already uses manifest-defined primary outputs.

The WATCHER implementation reads:

```python
primary_output = manifest.get("primary_output")
```

The current Step 3 manifest correctly declares:

```json
"primary_output": "survivors_with_scores.json"
```

The current Step 4 manifest correctly declares:

```json
"primary_output": "reinforcement_engine_config.json"
```

It explicitly states that Step 4 is a capacity planner, performs no evaluation, and uses file-existence validation rather than R².

Therefore:

```text
D1 appears resolved on current GitHub main.
D2 appears resolved on current GitHub main.
```

They should be removed from the list of current P1 defects unless the uncommitted Zeus working tree differs from GitHub `main`.

Alpha should compare:

```text
GitHub main
versus
the exact Zeus working tree used during S176
```

before retaining either defect.

---

# 7. Required baseline-acquisition sequence

## Phase A — Pre-run preparation

Before the first complete real-data cycle:

1. Add an explicit `BOOTSTRAP` governance state.
2. Disable autonomous enforcement of uncalibrated performance triggers.
3. Preserve active structural, file-integrity and catastrophic-failure gates.
4. Enable complete KPI recording.
5. Record the mathematical null rates.
6. Confirm every metric name has exactly one definition.
7. Confirm every policy value has a runtime consumer or is explicitly documentation-only.
8. Compare Zeus working-tree mappings against current GitHub `main`.

---

## Phase B — First complete real-data TFM cycle

The first cycle should verify:

```text
all stages execute
all expected artifacts are produced
WATCHER resolves correct files
diagnostics are calculated correctly
prediction pools have correct sizes
KPI history is persisted
trigger shadow decisions are recorded
```

This cycle produces initial observations.

It does not, by itself, establish a statistically meaningful baseline.

---

## Phase C — Causally correct historical walk-forward

Use historical data to produce pseudo-prospective observations:

```text
for each draw t:

    train only on information available before t

    run the required TFM pipeline stages

    generate prediction pools for t

    reveal actual draw t

    record:
        Hit@20
        Hit@100
        Hit@300
        pool metrics
        survivor metrics
        confidence metrics
        window metrics
        hypothetical trigger decisions

    advance to t+1
```

No data from draw `t` or later may enter:

* training;
* feature extraction;
* window selection;
* survivor scoring;
* threshold selection;
* prediction generation for draw `t`.

---

## Phase D — Candidate trigger simulation

Replay the historical KPI series against proposed policies.

For each candidate trigger, record:

```text
number of trigger fires
false-alarm rate
average time between fires
time to detect known degradation
trigger overlap
whether metric recovered without intervention
which action would have been requested
whether that action would have been proportionate
```

This converts threshold selection from guesswork into policy calibration.

---

## Phase E — Governed activation

Activate each trigger only after Team Beta verifies:

```text
sufficient observations
stable empirical distribution
acceptable false-alarm behavior
appropriate sensitivity
tested cooldown
tested hysteresis
clear action mapping
auditability
rollback safety
```

Triggers may transition individually rather than requiring all KPIs to become governed at once.

---

# 8. Analyzer review

The deterministic analyzer's central default arithmetic is correct.

Before it becomes a retained governance tool, Team Beta recommends:

1. Validate `pool_size > 0`.
2. Validate `draw_space > 0`.
3. Validate `max_misses >= 1`.
4. Do not hardcode `window_UNUSED=20`.
5. Do not hardcode `minimum_hit_rate=0.05`.
6. Read policy values from a supplied policy file or require them as explicit arguments.
7. Rename `chance_hit_probability` to `uniform_null_hit_probability`.
8. Keep `assumed_healthy_hit_rate` separate from the random null.
9. Assert or report the unique-pool-size assumption.
10. Replace `mean_gap > max_misses` as the general verdict criterion with a defined false-alarm horizon or expected waiting-time test.
11. Clarify that only two live triggers consume metric C; `minimum_hit_rate` is a configured target, not a live trigger.
12. Describe the Metric-A and Metric-C methods as complementary views of the same Bernoulli event rather than fully independent evidence sources.

These corrections do not alter the main S176 structural conclusion.

---

# 9. Final disposition

## Approved

* `current_hit_rate` is instantaneous rather than windowed.
* `hit_rate_collapse` currently re-encodes the latest miss.
* `hit_rate_collapse_window` is not used by the live trigger.
* `max_consecutive_misses=5` is not a defensible calibrated TFM degradation threshold.
* `minimum_hit_rate=0.05` is a synthetic-data target, not an empirical real-data baseline.
* Current outcome-space `pool_coverage` is not prospective predictive coverage.
* TFM presently has no empirical real-operation KPI baseline.
* WATCHER requires explicit `BOOTSTRAP`, `CALIBRATING`, and `GOVERNED` states.
* Historical walk-forward is required to bootstrap empirical baselines.
* Performance triggers should begin in audit-only mode.
* Structural and safety gates can remain active from the first cycle.
* Hit@K and pool-structure metrics must remain separate.

## Not approved

* Activating a replacement numerical hit-rate threshold now.
* Treating 5% as a validated minimum real-TFM hit rate.
* Treating the random 2% Top-20 null as healthy TFM performance.
* Activating the five-miss trigger as currently configured.
* Selecting a 20-draw collapse window without false-alarm analysis.
* Triggering retraining solely from low weight concentration.
* Claiming empirical TFM performance before walk-forward or prospective evidence exists.
* Listing D1 or D2 as current live-code defects without first showing that the Zeus working tree differs from current GitHub `main`.

---

# Final Team Beta ruling

> **S176 correctly identifies that WATCHER's active hit-collapse governance is structurally invalid for its stated purpose and that TFM has no empirical real-operation baseline from which performance thresholds can presently be calibrated. TFM must therefore begin in BOOTSTRAP mode: structural and catastrophic-failure gates remain active, mathematical null rates are recorded, and prospective performance KPIs operate in audit-only mode. A causally correct historical walk-forward must then establish empirical distributions and permit simulation of candidate trigger policies. Only triggers with acceptable false-alarm behavior, clear action mappings, cooldowns, hysteresis, auditability and rollback safety may enter autonomous GOVERNED operation. The separately reported Stage 3 and Stage 4 path defects are not present on current GitHub `main` and must be removed unless the S176 Zeus working tree is demonstrated to differ.**
