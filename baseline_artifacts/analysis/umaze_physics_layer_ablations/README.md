# UMaze physics-layer ablation: final report

This experiment tested whether explicitly representing motion magnitude improves a
DINOv2 latent world model, and whether speed and direction should be calibrated,
factorized, or regularized at probe-selected intermediate layers. Every planning
result below uses the epoch-20 checkpoint, seeds 100/200/300, and 50 evaluations
per seed (150 evaluations per condition).

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
