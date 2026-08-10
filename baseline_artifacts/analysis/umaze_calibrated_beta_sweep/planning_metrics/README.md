# UMaze planning evaluation: weak vs. strong calibrated speed

## Plain-English result

Increasing the calibrated-speed penalty by 100x did **not produce a clear
planning improvement**. The strong setting completed four more trials than the
weak setting (136/150 versus 132/150), but its average final state distance was
slightly worse. Both matched-run confidence intervals include zero, so this
experiment does not distinguish the two settings reliably.

The practical takeaway is that calibrated speed appears fairly robust across
these two extremes, but there is no evidence here that deliberately
overpowering it is better. If choosing between them from this evaluation alone,
prefer the weak setting because it is simpler and achieves statistically
indistinguishable behavior with less pressure on the representation.

## What was tested

The two trained conditions differ only in the weight on the existing
true-displacement calibrated-speed loss:

| Condition | Direction weight | Speed weight | beta |
|---|---:|---:|---:|
| `beta_0p1_weak` | 0.1 | 0.01 | 0.1 |
| `beta_10_strong` | 0.1 | 1.0 | 10 |

Each checkpoint was evaluated on the same 150 planning trials: three seeds
(100, 200, 300), five matched 10-trial chunks per seed, and identical chunk
offsets (0, 10, 20, 30, 40). This matching lets us compare strong minus weak
within the same seed/offset block rather than treating unrelated runs as pairs.

## The two new tests

### 1. Success rate

A trial succeeds when the agent's final **XY position** is within 0.5 units of
the goal. This is the most direct answer to “did the planner reach the goal?”

### 2. Final state distance

The logged `final_eval/mean_state_dist` is the Euclidean distance between the
final and goal **four-dimensional state** (XY position plus velocity). Lower is
better. Despite the convenient phrase “absolute distance to goal,” this is not
pure XY distance: a trial can be positionally successful while still receiving
a nonzero distance because its terminal velocity differs from the goal state.

## Results

| Condition | Successes | Success rate | 95% interval | Mean final state distance | 95% interval |
|---|---:|---:|---:|---:|---:|
| Weak beta 0.1 | 132/150 | 88.0% | 82.5–93.5% | 3.216 | 2.845–3.586 |
| Strong beta 10 | 136/150 | 90.7% | 86.2–95.1% | 3.270 | 2.995–3.546 |

The intervals above use the 15 chunk-level measurements. Every chunk contains
10 trials, so the unweighted chunk mean is also the trial-weighted mean.

### Matched strong-minus-weak comparison

| Metric | Mean delta | 95% interval | Better strong chunks | Interpretation |
|---|---:|---:|---:|---|
| Success rate | +2.67 percentage points | -3.22 to +8.55 pp | 6/15 (4 ties) | Direction favors strong, but the interval spans worse and better outcomes. |
| Final state distance | +0.055 | -0.290 to +0.399 | 8/15 lower | Mean direction slightly favors weak; chunk wins are nearly split. |

The apparent success advantage is only four trials and is not consistent
across seeds: strong is better for seed 100, weak is better for seed 200, and
strong is modestly better for seed 300. That instability is another reason not
to treat the aggregate difference as a decisive beta effect.

| Seed | Weak success | Strong success | Weak distance | Strong distance |
|---:|---:|---:|---:|---:|
| 100 | 90% | 94% | 2.858 | 3.386 |
| 200 | 94% | 88% | 3.378 | 3.183 |
| 300 | 80% | 90% | 3.411 | 3.242 |

## Interpretation alongside the Spearman test

The Spearman analysis asks whether latent distances preserve A* path-length
ordering. These two tests instead ask whether the learned model actually plans
to the goal and where it finishes. A representation can improve a geometric
correlation without improving closed-loop planning, so behavioral metrics
should be the decision criterion and Spearman should be treated as diagnostic.

Here, the behavioral comparison says the strong penalty did not buy a reliable
gain. Its thresholded XY success is slightly higher, while its full-state
distance is slightly worse; both differences are small relative to run-to-run
variation.

## Limitations and next checks

- Only beta 0.1 and beta 10 have these planning results. Run the same matched
  evaluation for R0 and the beta 1 reference before selecting a final model.
- Per-trial final distances were not retained, so distance uncertainty is based
  on 15 chunk means rather than all 150 individual outcomes.
- Add an explicit `mean_xy_dist` metric if the intended question is purely
  positional distance to the goal; the current distance also includes velocity.
- Repeat with additional training seeds to separate beta sensitivity from
  checkpoint-specific variance.

Machine-readable aggregates are in [`summary.json`](summary.json).
