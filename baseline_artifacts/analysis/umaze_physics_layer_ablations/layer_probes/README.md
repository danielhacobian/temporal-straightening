# UMaze layerwise physics probes

These probes ask what information is *linearly readable* from each representation. They do not by themselves prove causal use by the planner.

- Windows: 512; frame skip: 5; split is by window.
- Collision/stall means above-median commanded action with bottom-15% physical displacement.
- `action_to_latent_r2` fits a local action→latent-change map; its column space is the action-controllable subspace.
- `astar_action_subspace_overlap` is the squared principal-cosine overlap between the A* probe and that controllable subspace.
- Layer selection averages non-negative A* rank correlation, speed R², direction cosine, action decode R², and collision AUC above chance.

## Selected layers

- DINO: layer 6 (pooled_patches, score 0.381)
- Predictor: layer 1 (pooled_visual, score 0.331)

## Highest-scoring representations

| representation | score | A* ρ | speed R² | direction cos | collision AUC | action R² | A*↔control overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| dino/6/pooled_patches | 0.381 | 0.995 | -0.249 | 0.909 | 0.392 | -0.123 | 0.004 |
| dino/5/pooled_patches | 0.380 | 0.994 | -0.178 | 0.906 | 0.393 | -0.081 | 0.009 |
| dino/4/pooled_patches | 0.378 | 0.995 | -0.162 | 0.897 | 0.409 | -0.097 | 0.001 |
| dino/8/pooled_patches | 0.378 | 0.993 | -0.140 | 0.896 | 0.418 | -0.068 | 0.004 |
| dino/10/pooled_patches | 0.376 | 0.993 | -0.189 | 0.888 | 0.390 | -0.118 | 0.005 |
| dino/7/pooled_patches | 0.375 | 0.993 | -0.190 | 0.883 | 0.354 | -0.131 | 0.002 |
| dino/11/projected_aggregate | 0.375 | 0.982 | -0.029 | 0.853 | 0.503 | 0.035 | 0.015 |
| dino/9/pooled_patches | 0.375 | 0.993 | -0.153 | 0.881 | 0.430 | -0.103 | 0.001 |
| dino/11/pooled_patches | 0.375 | 0.992 | -0.117 | 0.881 | 0.447 | -0.073 | 0.001 |
| dino/6/cls | 0.374 | 0.988 | -0.128 | 0.882 | 0.403 | -0.047 | 0.002 |
| dino/3/pooled_patches | 0.373 | 0.994 | -0.212 | 0.869 | 0.360 | -0.117 | 0.002 |
| dino/7/cls | 0.372 | 0.991 | -0.110 | 0.869 | 0.453 | -0.037 | 0.000 |

## Interpretation

A high A* score means latent ordering agrees with true maze progress. A high speed or direction score means physical motion is easy to recover. High action→latent R² means commands move the representation predictably. Overlap tells us whether maze-distance information lies in directions the action-conditioned predictor can actually manipulate; overlap that is too low suggests planning-relevant geometry is present but dynamically inaccessible.

The selected layers are the inputs to the layer-aware factorized ablation. The full table remains the primary result; selection should be treated as a preregistered heuristic rather than post-hoc proof.
