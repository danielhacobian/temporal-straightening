# UMaze physics-layer ablation: final report

This experiment tested whether explicitly representing motion magnitude improves a
DINOv2 latent world model, and whether speed and direction should be calibrated,
factorized, or regularized at probe-selected intermediate layers. Every planning
result below uses the epoch-20 checkpoint, seeds 100/200/300, and 50 evaluations
per seed (150 evaluations per condition).

## Plain-English overview

### The experiment in one sentence

We tested whether explicitly teaching a visual world model about motion--especially
speed and direction--would make its internal map of UMaze more physically honest
and therefore improve planning.

The hypothesis was:

> Better motion representation -> more honest latent distances -> better plans.

### What problem were we investigating?

The model sees UMaze as images and converts every image into a latent
representation: a numerical description of the current situation. The planner
uses distances between these representations to choose a route to the goal.

Ideally, nearby latent states should be easy to travel between, distant states
should require longer routes, and states separated by a wall should not look
deceptively close. A latent step should also reflect both **where** the agent moved
and **how much** it moved.

The concern was that a model could recognize the visual appearance of each
location without learning the maze's real movement geometry. If that happens,
the planner can be given a visually plausible but physically misleading map.

### Step 1: find where the model stores physical information

Before changing training, we froze the existing model and trained simple linear
probes at every DINO and predictor layer. A linear probe asks whether a physical
quantity can be recovered from a representation using only a simple linear
equation. It does not change the world model.

We probed for:

- absolute and goal-relative position;
- Cartesian velocity `(vx, vy)` and acceleration;
- polar speed and heading;
- collision;
- action magnitude; and
- true A* distance to the goal.

We tested CLS features, individual patches, pooled patches, projected features,
and intermediate predictor features. We also compared Cartesian velocity with a
polar description that explicitly separates speed from direction:

```text
speed = sqrt(vx^2 + vy^2)
heading = atan2(vy, vx)
```

Position was already extremely easy to decode from frozen DINO features
(approximately 0.99 R²). Speed and direction frequently occupied nearly
orthogonal subspaces, with angles around 83--85 degrees. The probe sweep selected
DINO layer 6 pooled patches and predictor layer 1 pooled visual features.

The selection score tells us where information is easy to **read**. It does not
prove that applying a training loss at that layer will help planning.

### Step 2: train five versions of the model

- **R0** is the direction-only baseline. It encourages consecutive latent changes
  to point consistently, but does not directly constrain latent step size.
- **R2** constrains direction and latent step magnitude. In simple terms, it asks
  the model to move through latent space at a steadier pace.
- **Calibrated speed** adds a target based on the agent's true physical
  displacement. Larger real movements should cause appropriately larger latent
  movements instead of treating every transition as equally large.
- **Factorized** gives direction and speed separate learned projections, with the
  goal of preventing the two signals from interfering.
- **Layer-aware factorized** applies those separate losses at the two
  probe-selected intermediate layers instead of only at the final planning space.

### Step 3: test whether each model can actually plan

Every condition received the same matched planning evaluation: three seeds, 50
evaluations per seed, and 150 evaluations per condition. The primary metric was
the percentage of trials that reached the goal. We also measured final
proprioceptive, state, and visual distance.

R2 was the best planner at 94.00% success. Calibrated speed reached 93.33%, and
both beat the 92.00% R0 baseline. Factorized fell to 86.67%, while layer-aware
factorized collapsed to 24.00%.

### Step 4: test whether latent distance became more honest

We placed the agent on a grid of valid UMaze states. For every state, we compared:

1. its visual latent distance to the goal; and
2. the true shortest-path distance to the goal computed by A*.

A* respects the maze walls, so it measures how far the goal really is along a
traversable route. Spearman correlation then measures whether the latent model
ranks states in the same near-to-far order as A*.

Calibrated speed produced the best correlation: 0.9791 compared with 0.9560 for
R0. This directly supports the proposed middle link: calibrating latent speed to
true movement made the internal distance map more faithful to the maze.

Factorized scored 0.3152 and layer-aware factorized scored 0.6259. Their global
latent geometry became much less faithful, matching their weaker planning.

### Step 5: check what motion information remained readable

After training, we repeated detailed Cartesian and polar motion probes. Some of
the unsuccessful models made local physical variables *more* readable. For
example, factorized had the most decodable speed, while layer-aware factorized
had the most decodable velocity. Yet both planned worse than R0.

This is the experiment's most important warning:

> A model can store motion information clearly while still arranging its latent
> map badly for planning.

Linear probes measure whether information exists and is easy to extract. A
planner additionally needs obstacle topology, controllability, and goal-distance
ordering to remain coherent in the same latent space.

### Bottom line

The speed term is doing something useful; it is not merely an extra constraint.
R2 gave the best immediate planning result, while calibrated speed gave the
strongest evidence that physically meaningful speed supervision makes latent
distance more honest.

Naively separating speed and direction, especially at intermediate layers, was
harmful. The safest next direction is therefore to keep R2 or calibrated speed in
the final planner representation and use layer probes as diagnostics rather than
automatic instructions for where to apply a loss.

## Conditions

- **R0:** existing direction-only trajectory regularizer.
- **R2:** existing scale-invariant direction-plus-constant-speed penalty.
- **Calibrated speed:** R0 plus speed calibration against true displacement.
- **Factorized:** separate learned direction and speed projections.
- **Layer-aware factorized:** factorized penalties applied at probe-selected
  DINO and predictor layers.

## Planning results

Higher success is better; lower distance is better. Distances are reported at the
final evaluation step.

| Condition | Success | Delta vs R0 | Proprio distance | State distance | Visual distance |
|---|---:|---:|---:|---:|---:|
| R0 | 92.00% | baseline | 0.9521 | 2.8816 | 0.3905 |
| R2 | **94.00%** | **+2.00 pp** | 0.9615 | 2.9215 | 0.3984 |
| Calibrated speed | 93.33% | +1.33 pp | 0.9578 | 2.9343 | 0.3963 |
| Factorized | 86.67% | -5.33 pp | 1.0683 | 3.1875 | 0.4035 |
| Layer-aware factorized | 24.00% | -68.00 pp | 1.3755 | 3.4610 | 0.5208 |

R2 is the best planner in this run, while calibrated speed preserves the small
gain over R0 but does not surpass R2. Both factorized variants hurt planning,
especially the layer-aware version.

## Does latent distance match true path distance?

For each valid grid state, visual latent distance to the canonical goal was
compared with the true A* step-count. Spearman correlation measures whether the
latent representation orders states by actual maze distance.

| Comparison | R0 Spearman rho | Treatment rho | Paired delta |
|---|---:|---:|---:|
| R0 vs calibrated speed | 0.9560 | **0.9791** | **+0.0231** |
| R0 vs factorized | 0.9560 | 0.3152 | -0.6408 |
| R0 vs layer-aware factorized | 0.9560 | 0.6259 | -0.3301 |

Calibrated speed strongly supports the proposed middle link: its latent distances
are more faithful to true maze distance. The improvement is small because R0 is
already very strong, but it is consistent with calibrated speed's retained
planning success. The factorized models show the opposite pattern: their global
latent geometry is much less faithful, matching their worse planning.

## Layer probes

The initial linear probes selected:

- **DINO layer 6**, pooled patches, selection score **0.381**.
- **Predictor layer 1**, pooled visual features, selection score **0.331**.

The selection score averages non-negative A* rank correlation, speed R2,
direction cosine, action-decode R2, and collision AUC above chance. It measures
linear readability, not causal usefulness to planning. Frozen DINO pooled patches
encoded position extremely well (approximately 0.99 R2), and speed/direction
subspaces were nearly orthogonal at many DINO representations (roughly 83--85
degrees).

## Motion-geometry probes

The table shows representative predictor-layer probes. Higher values mean the
quantity is more linearly recoverable.

| Condition | Velocity R2 | Speed R2 | Heading cosine | Acceleration R2 | Acceleration-direction cosine |
|---|---:|---:|---:|---:|---:|
| R0 | 0.108 | 0.053 | 0.278 | 0.111 | 0.282 |
| R2 | 0.250 | 0.107 | **0.480** | 0.061 | 0.215 |
| Calibrated speed | 0.176 | 0.062 | 0.335 | 0.091 | 0.273 |
| Factorized | 0.233 | **0.163** | 0.412 | 0.102 | 0.250 |
| Layer-aware factorized | **0.290** | 0.009 | 0.454 | **0.160** | **0.320** |

These probes expose an important distinction. Factorized and layer-aware models
can make some local motion variables easier to decode while simultaneously
destroying the global geometry required by planning. Linear decodability is
therefore not enough: the useful representation must preserve controllability,
obstacle topology, and goal-distance ordering in the same space used by the
planner.

## Interpretation

1. **The speed term is useful, but the simple R2 form is the strongest tested
   implementation.** It gives the highest success rate and substantially improves
   velocity/heading readability.
2. **True-displacement calibration improves geometric honesty.** It produces the
   best A* correlation and slightly improves success over R0, directly supporting
   the hypothesized middle link.
3. **Naive factorization is harmful.** Separating motion variables without
   preserving planner-compatible global geometry produces readable local physics
   but worse control.
4. **Probe-selected layers are not automatically good regularization sites.** The
   severe layer-aware failure shows that a layer-selection objective must include
   downstream geometry or planning sensitivity, not only linear readability.
5. **The next minimal experiment should retain R2/calibrated speed in the final
   planner space** and use probes only as diagnostics. If intermediate losses are
   retried, introduce them weakly and sweep their coefficient while monitoring A*
   correlation before committing to full planning.

## Artifact layout

- `layer_probes/`: original layerwise probes.
- `motion_geometry_probes/`: Cartesian/polar motion probes, subspace angles, and
  cross-patch redundancy analyses.
- `geodesic_r0_vs_calibrated_speed/`: R0/calibrated A* comparison.
- `geodesic_r0_vs_factorized/`: R0/factorized A* comparison.
- `geodesic_r0_vs_layer_aware_factorized/`: R0/layer-aware A* comparison.
- `../../plans/umaze_physics_layer_ablations/comparison.json`: complete planning
  aggregation and paired deltas.

Persistent object-storage prefix:
`s3://temporal-straightening/umaze_physics_layer_ablations/`.
