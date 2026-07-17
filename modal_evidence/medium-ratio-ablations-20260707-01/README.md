# Medium Ratio-Speed Ablations, 2026-07-07

This folder contains compact local evidence for testing Erin's scale-invariant speed hypothesis on the Medium environment. Both runs used the trainable `dino_channel` adapter path, 50 random-policy Medium episodes of length 100, 20 training epochs, and gradient-descent planning with 50 evaluations and goal horizon 25.

## Runs

| Variant | Loss | Train | Val | Success | State dist | Curvature | Ratio penalty | Norm. accel | Speed CV | Rollouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dino_channel_ratio_speed` | `ratiospeed1e-1` | 0.1426 | 0.1408 | 0.12 | 4.2253 | 1.3435 | 0.3153 | 3.0024 | 0.3793 | 20 |
| `dino_channel_normacc` | `normacc1e-1` | 0.3688 | 0.3339 | 0.16 | 3.5665 | 1.0340 | 0.5058 | 2.5737 | 0.4510 | 20 |

## Interpretation

The ratio-speed term did what it was designed to do: it avoids directly shrinking latent step magnitudes and produced the lowest ratio-speed penalty among the two new variants. Compared with the earlier magnitude speed-only adapter ablation, it kept much larger latent step speeds, so it does not look like the same smoothness-collapse behavior.

Ratio-speed alone did not make the latent path directionally straighter and did not improve planner success beyond the prior cosine adapter. The normalized-acceleration term, which combines ratio speed and cosine curvature as Erin described, reduced curvature and full normalized acceleration and produced the best adapter-family success rate so far, 0.16. It did not beat the older DINO CLS baseline at 0.22 and it did not make speed metrics best-in-class, so this is evidence for a meaningful direction, not a final claim.

## Evidence Files

- `pulled/medium-ratio-speed-summary.json`: training and planning summary for `ratiospeed1e-1`.
- `pulled/medium-ratio-speed-latent_analysis.json`: 20-rollout latent diagnostics for `ratiospeed1e-1`.
- `pulled/medium-ratio-speed-plan-logs.json`: final planner JSONL metrics for `ratiospeed1e-1`.
- `raw_volume/medium-ratio-speed-logs`: training command/log with `Ratio speed constancy enabled: mode=cos, scale=0.1`.
- `pulled/medium-normacc-summary.json`: training and planning summary for `normacc1e-1`.
- `pulled/medium-normacc-latent_analysis.json`: 20-rollout latent diagnostics for `normacc1e-1`.
- `pulled/medium-normacc-plan-logs.json`: final planner JSONL metrics for `normacc1e-1`.
- `raw_volume/medium-normacc-logs`: training command/log with `Normalized acceleration enabled: mode=cos, scale=0.1`.

The 50-rollout ratio latent diagnostic was aborted after it behaved like the earlier stalled adapter speed diagnostic. Both new variants were therefore compared with the same 20-rollout latent cap.

## Modal Runs

- Ratio-speed training/planning: https://modal.com/apps/usharma123/main/ap-h4o9eT5603aOfUiE3bn16p
- Ratio-speed 20-rollout latent analysis: https://modal.com/apps/usharma123/main/ap-JhNefasMJyD9ejbcGdzNfs
- Normalized-acceleration training/planning: https://modal.com/apps/usharma123/main/ap-EnYiaEETorgs1O5wKIiq3X
- Normalized-acceleration 20-rollout latent analysis: https://modal.com/apps/usharma123/main/ap-qNWjH37WurYOpcMKabXUn8

