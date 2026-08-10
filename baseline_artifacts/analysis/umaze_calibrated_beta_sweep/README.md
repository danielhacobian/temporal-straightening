# UMaze calibrated-speed beta sweep

This sweep holds the existing R0 direction regularizer fixed at 0.1 and varies
only the true-displacement calibrated speed term.

| Condition | Direction weight | Speed weight | beta = speed / direction |
|---|---:|---:|---:|
| Weak | 0.1 | 0.01 | 0.1 |
| Existing calibrated reference | 0.1 | 0.1 | 1 |
| Strong / deliberately overpowered | 0.1 | 1.0 | 10 |

The speed loss uses the existing implementation: for each adjacent frame,
`latent_speed = ||z[t+1]-z[t]||`, `physical_speed = ||s[t+1,:2]-s[t,:2]||`,
and `r[t] = log(latent_speed) - log(physical_speed)`. The penalty is smooth-L1
on `r[t] - stop_gradient(mean(r))`, so it calibrates relative latent pace to
true displacement without fixing an arbitrary global latent scale.

Training uses the same UMaze dataset, seed, architecture, 20 epochs, batch size
32, direction weight, and optimizer settings as the existing calibrated-speed
experiment. Only the speed weight changes.
