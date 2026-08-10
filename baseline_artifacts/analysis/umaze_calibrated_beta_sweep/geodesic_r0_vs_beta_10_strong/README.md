# UMaze latent-geodesic correlation: R0 vs R2

This analysis tests the missing middle link in the hypothesis:

> steadier latent pace → more honest latent distances → better planning

The visual latent Euclidean distance from each valid grid state to the
canonical UMaze goal was compared with true A* step-count. The same rendered
observations were passed through both checkpoints. Agent velocity was fixed at
zero and the planning-matched visual representation was used (`alpha=0`).

| Condition | Spearman ρ | 95% bootstrap CI | p-value |
|---|---:|---:|---:|
| R0 direction-only | 0.9472 | [0.9289, 0.9608] | 9.03e-233 |
| R2 full penalty | 0.9054 | [0.8767, 0.9282] | 6.83e-176 |

Paired difference, `ρ(R2) - ρ(R0)`: **-0.0417**, with a paired state
bootstrap 95% CI of **[-0.0545, -0.0311]**.

This result **does not support** the proposed middle link. A positive,
well-separated difference means R2 orders states by true path distance more
faithfully than R0. This is a geometry diagnostic, not by itself evidence that
the improvement causes planning success.

Artifacts:

- `grid_latent_distances.csv`: paired per-state measurements
- `results.json`: machine-readable statistics and run settings
- `latent_vs_astar_scatter.png`: correlation view
- `distance_fields.png`: true and learned distance maps
