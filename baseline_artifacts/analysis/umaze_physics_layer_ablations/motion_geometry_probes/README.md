# UMaze Cartesian/polar motion-geometry probes

This post-training study tests whether physical variables are linearly readable, not merely correlated with planning success. All probes use the same held-out trajectory-window split and the same ridge penalty.

## Questions

1. Is position readable as `(x,y)`, and is goal-relative position cleaner as range/bearing?
2. Is velocity cleaner as Cartesian `(vx,vy)` or polar `(speed, heading)`?
3. Does acceleration become readable at the same layer as velocity?
4. Are direction and speed/other intuitive-physics probe subspaces close to orthogonal?
5. How many linearly independent heading harmonics are supported, and is heading spatially redundant across patches?

Direction is represented as `(cos θ, sin θ)`, avoiding the discontinuity at ±π. Acceleration is the finite difference of the dataset's physical velocity. Direction metrics exclude the slowest 10% of samples, where angle is ill-defined.

## Selected-layer checkpoint comparison

| condition | representation | pos R² | vel R² | speed R² | heading cos | accel R² | accel cos | speed↔heading overlap | min angle | heading d90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r0 | dino/6/cls | 0.990 | -0.134 | 0.008 | 0.024 | 0.064 | 0.284 | 0.009 | 84.6° | 26 |
| r0 | dino/6/individual_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| r0 | dino/6/pooled_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| r0 | dino/6/projected_aggregate | 0.958 | -0.030 | 0.078 | 0.085 | 0.190 | 0.383 | 0.008 | 84.9° | 19 |
| r0 | dino/6/projected_patches | 0.772 | -0.022 | 0.031 | 0.063 | 0.111 | 0.271 | 0.654 | 36.0° | 3 |
| r0 | predictor/1/individual_visual_tokens | 0.708 | 0.108 | 0.053 | 0.278 | 0.111 | 0.282 | 0.136 | 68.4° | 4 |
| r0 | predictor/1/pooled_visual | 0.708 | 0.108 | 0.053 | 0.278 | 0.111 | 0.282 | 0.136 | 68.4° | 4 |
| r2 | dino/6/cls | 0.990 | -0.134 | 0.008 | 0.024 | 0.064 | 0.284 | 0.009 | 84.6° | 26 |
| r2 | dino/6/individual_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| r2 | dino/6/pooled_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| r2 | dino/6/projected_aggregate | 0.983 | -0.066 | 0.060 | 0.035 | 0.163 | 0.319 | 0.020 | 81.8° | 23 |
| r2 | dino/6/projected_patches | 0.658 | -0.007 | 0.017 | 0.052 | 0.120 | 0.312 | 0.079 | 73.7° | 4 |
| r2 | predictor/1/individual_visual_tokens | 0.632 | 0.250 | 0.107 | 0.480 | 0.061 | 0.215 | 0.320 | 55.6° | 3 |
| r2 | predictor/1/pooled_visual | 0.632 | 0.250 | 0.107 | 0.480 | 0.061 | 0.215 | 0.320 | 55.6° | 3 |
| calibrated_speed | dino/6/cls | 0.990 | -0.134 | 0.008 | 0.024 | 0.064 | 0.284 | 0.009 | 84.6° | 26 |
| calibrated_speed | dino/6/individual_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| calibrated_speed | dino/6/pooled_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| calibrated_speed | dino/6/projected_aggregate | 0.954 | -0.026 | 0.069 | 0.081 | 0.175 | 0.364 | 0.037 | 78.9° | 20 |
| calibrated_speed | dino/6/projected_patches | 0.860 | -0.020 | 0.035 | 0.057 | 0.119 | 0.262 | 0.421 | 49.6° | 4 |
| calibrated_speed | predictor/1/individual_visual_tokens | 0.620 | 0.176 | 0.062 | 0.335 | 0.091 | 0.273 | 0.351 | 53.7° | 4 |
| calibrated_speed | predictor/1/pooled_visual | 0.620 | 0.176 | 0.062 | 0.335 | 0.091 | 0.273 | 0.351 | 53.7° | 4 |
| factorized | dino/6/cls | 0.990 | -0.134 | 0.008 | 0.024 | 0.064 | 0.284 | 0.009 | 84.6° | 26 |
| factorized | dino/6/individual_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| factorized | dino/6/pooled_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| factorized | dino/6/projected_aggregate | 0.987 | -0.059 | 0.080 | 0.042 | 0.149 | 0.326 | 0.035 | 79.3° | 23 |
| factorized | dino/6/projected_patches | 0.544 | -0.012 | 0.075 | 0.037 | 0.075 | 0.214 | 0.231 | 61.3° | 4 |
| factorized | predictor/1/individual_visual_tokens | 0.655 | 0.233 | 0.163 | 0.412 | 0.102 | 0.250 | 0.314 | 55.9° | 3 |
| factorized | predictor/1/pooled_visual | 0.655 | 0.233 | 0.163 | 0.412 | 0.102 | 0.250 | 0.314 | 55.9° | 3 |
| layer_aware_factorized | dino/6/cls | 0.990 | -0.134 | 0.008 | 0.024 | 0.064 | 0.284 | 0.009 | 84.6° | 26 |
| layer_aware_factorized | dino/6/individual_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| layer_aware_factorized | dino/6/pooled_patches | 0.995 | -0.173 | -0.041 | 0.019 | 0.032 | 0.274 | 0.015 | 83.1° | 27 |
| layer_aware_factorized | dino/6/projected_aggregate | 0.981 | -0.068 | 0.041 | 0.038 | 0.148 | 0.360 | 0.021 | 81.6° | 22 |
| layer_aware_factorized | dino/6/projected_patches | 0.381 | -0.016 | 0.030 | -0.016 | 0.033 | 0.108 | 0.359 | 53.2° | 5 |
| layer_aware_factorized | predictor/1/individual_visual_tokens | 0.999 | 0.290 | 0.009 | 0.454 | 0.160 | 0.320 | 0.303 | 56.6° | 3 |
| layer_aware_factorized | predictor/1/pooled_visual | 0.999 | 0.290 | 0.009 | 0.454 | 0.160 | 0.320 | 0.303 | 56.6° | 3 |

## Interesting findings

### 1. DINO is an excellent position encoder but a weak direct motion encoder

At DINO layer 6, pooled patches decode position with R² 0.995, but Cartesian
velocity and speed have negative held-out R² values (-0.173 and -0.041). This
means the frozen visual features identify *where* the agent is extremely well,
while instantaneous motion is not directly readable with a simple linear map.

The DINO rows are identical across conditions because the DINO encoder was held
fixed. Most condition-dependent changes therefore appear in projected and
predictor representations rather than the underlying visual backbone.

### 2. Speed and heading use nearly orthogonal directions in selected DINO features

For R0 at DINO layer 6, the speed-versus-heading probe-weight overlap is only
0.009--0.015 for CLS, individual patches, pooled patches, and projected aggregate
features. Their minimum principal angles are 83.1--84.9 degrees; 90 degrees would
be perfectly orthogonal.

This result specifically compares the scalar **speed** probe with the
`(cos heading, sin heading)` **direction** probe. It should not be described as
direction being orthogonal to the complete velocity vector, because velocity
already contains both magnitude and direction.

### 3. The predictor compresses and combines motion information

At the selected predictor layer, the speed-versus-heading angle drops to
53.7--68.4 degrees and overlap rises to 0.136--0.351. Heading also needs only
3--4 singular directions to explain 90% of the harmonic-probe weight energy,
compared with roughly 19--27 directions in the selected DINO representations.

The predictor therefore appears to compress a distributed DINO direction code
into a smaller, more entangled motion representation. This is an association in
the probe weights, not proof that the planner causally uses those exact axes.

### 4. R2 produces the strongest balanced predictor-level motion readout

Relative to R0 at `predictor/1/pooled_visual`, R2 improves:

- velocity R² from 0.108 to 0.250;
- speed R² from 0.053 to 0.107; and
- heading cosine from 0.278 to 0.480.

R2 also achieved the best planning success, 94% versus 92% for R0. Among the
tested conditions, it gives the clearest alignment between improved local motion
readability and improved downstream planning.

### 5. The easiest-to-decode speed is not the best speed representation for planning

Factorized training gives the highest predictor speed R², 0.163, and improves
heading cosine to 0.412. Nevertheless, its planning success falls to 86.67%, and
its latent/A* Spearman correlation collapses to 0.3152 from R0's 0.9560.

This is direct evidence that linear decodability is not sufficient. Speed can be
stored in a clean linear direction while the global state geometry needed for
obstacle-aware planning is damaged.

### 6. Layer-aware factorization creates the strongest local/global contradiction

Layer-aware factorized has the highest predictor position R² (0.999), velocity
R² (0.290), acceleration R² (0.160), and acceleration-direction cosine (0.320).
It also retains a strong heading cosine of 0.454. Yet its speed R² is only 0.009,
planning success collapses to 24%, and latent/A* correlation falls to 0.6259.

The representation is locally informative but globally unsuitable for planning.
This is the clearest warning against selecting regularization sites solely from
linear probe scores.

### 7. Calibrated speed improves global geometry more than local probe scores suggest

Calibrated speed has only moderate predictor motion scores: velocity R² 0.176,
speed R² 0.062, and heading cosine 0.335. However, it produces the best
latent-versus-A* correlation, 0.9791, and improves planning success to 93.33%.

This suggests that calibration's main benefit is not making speed maximally easy
to decode. Instead, it makes distances across the complete latent map more
consistent with true traversable distance.

### Overall takeaway

The probes reveal two distinct properties:

1. **Local readability:** whether position, speed, heading, or acceleration can be
   extracted with a linear model.
2. **Global planning geometry:** whether the full latent space preserves walls,
   controllability, and correct goal-distance ordering.

R2 improves both enough to give the best planning result. Calibrated speed mainly
improves global distance honesty. The factorized variants show that improving
local readability alone can actively harm planning.

## Reading the results

- Higher R²/cosine and lower RMSE/angular error mean easier linear decoding.
- Polar dominance means strong speed R² and heading cosine even when joint Cartesian velocity R² is weaker.
- Similar velocity and acceleration layer-onset supports the paper's claim that acceleration can emerge without a separate intermediate velocity stage.
- Low subspace overlap and a large principal angle mean the two variables occupy distinct directions in feature space.
- `heading_harmonic_dimensions_90pct` is the number of singular directions needed for 90% of a 32-target circular-harmonic probe's weight energy; it is a diagnostic of distributed direction coding, not the intrinsic dimension of the complete representation.
- High pairwise patch prediction cosine and many patches near the best patch indicate spatially redundant heading information.

These are diagnostic associations. A variable being decodable does not prove that the predictor or planner causally uses it.
