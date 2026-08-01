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

## Reading the results

- Higher R²/cosine and lower RMSE/angular error mean easier linear decoding.
- Polar dominance means strong speed R² and heading cosine even when joint Cartesian velocity R² is weaker.
- Similar velocity and acceleration layer-onset supports the paper's claim that acceleration can emerge without a separate intermediate velocity stage.
- Low subspace overlap and a large principal angle mean the two variables occupy distinct directions in feature space.
- `heading_harmonic_dimensions_90pct` is the number of singular directions needed for 90% of a 32-target circular-harmonic probe's weight energy; it is a diagnostic of distributed direction coding, not the intrinsic dimension of the complete representation.
- High pairwise patch prediction cosine and many patches near the best patch indicate spatially redundant heading information.

These are diagnostic associations. A variable being decodable does not prove that the predictor or planner causally uses it.
