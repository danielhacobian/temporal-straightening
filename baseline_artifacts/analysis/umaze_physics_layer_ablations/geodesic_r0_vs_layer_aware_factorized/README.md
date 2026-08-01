# UMaze latent-geodesic correlation: R0 vs R2

This analysis tests the missing middle link in the hypothesis:

> steadier latent pace → more honest latent distances → better planning

The visual latent Euclidean distance from each valid grid state to the
canonical UMaze goal was compared with true A* step-count. The same rendered
observations were passed through both checkpoints. Agent velocity was fixed at
zero and the planning-matched visual representation was used (`alpha=0`).

| Condition | Spearman ρ | 95% bootstrap CI | p-value |
|---|---:|---:|---:|
| R0 direction-only | 0.9560 | [0.9401, 0.9679] | 7.01e-251 |
| R2 full penalty | 0.6259 | [0.5471, 0.6948] | 2.29e-52 |

Paired difference, `ρ(R2) - ρ(R0)`: **-0.3301**, with a paired state
bootstrap 95% CI of **[-0.3997, -0.2678]**.

This result **does not support** the proposed middle link. A positive,
well-separated difference means R2 orders states by true path distance more
faithfully than R0. This is a geometry diagnostic, not by itself evidence that
the improvement causes planning success.

Artifacts:

- `grid_latent_distances.csv`: paired per-state measurements
- `results.json`: machine-readable statistics and run settings
- `latent_vs_astar_scatter.png`: correlation view
- `distance_fields.png`: true and learned distance maps
