# Temporal Straightening Reproduction Report

Date: 2026-07-03

## Executive Summary

I reproduced the Medium environment baseline with the authors' temporal straightening setup and DINO baselines, ran speed-constancy ablations, pulled local evidence, and added code to make the Wall environment runnable. The completed Medium results show that DINO CLS was stronger than the patch-token variants in this setup, while frozen DINO patch variants had identical latent diagnostics and identical image evidence under speed ablations.

The main caveat in the first ablation is real: with frozen DINO features, speed losses cannot change the measured frozen latent geometry. I started a more meaningful trainable-adapter ablation using `encoder=dino_channel`, where the DINO backbone remains frozen but the measured projection path is trainable. Those runs produced partial training evidence, but Modal then hit `workspace billing cycle spend limit reached`, preventing planner completion, latent diagnostics, Wall smoke, and Wall full runs.

## What Was Run

Medium completed:

| Variant | Encoder | Loss setting | Epoch | Train loss | Val loss | Planner success | Mean state dist |
|---|---:|---:|---:|---:|---:|---:|---:|
| Authors approach | DINO patch | `cos1e-1` | 20 | 0.2552 | 0.2385 | 0.14 | 4.4943 |
| DINO patch | DINO patch | `False` | 20 | 0.1195 | 0.1025 | 0.14 | 4.4943 |
| DINO CLS | DINO CLS | `False` | 20 | 0.0785 | 0.0691 | 0.22 | 3.5456 |

Frozen-DINO speed ablations completed:

| Variant | Encoder | Loss setting | Epoch | Train loss | Val loss | Planner success | Main interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Patch speed | DINO patch | `aggspeed1e-1` | 20 | 0.1268 | 0.1094 | 0.14 | No planner/latent change versus frozen patch baseline |
| Patch cosine + speed | DINO patch | `cos1e-1+aggspeed1e-1` | 20 | 0.2625 | 0.2455 | 0.14 | Same frozen patch geometry as baseline |
| CLS speed | DINO CLS | `speed1e-1` | 20 | 0.0860 | 0.0754 | 0.22 | Same CLS geometry as CLS baseline |

Trainable-adapter ablation status:

| Variant | Loss setting | Last epoch | Train loss | Val loss | Planner status |
|---|---:|---:|---:|---:|---|
| `dino_channel_no_straightening` | `False` | 20 | 0.1064 | 0.0957 | Started, stopped at step 1/1000 |
| `dino_channel_cos_straightened` | `cos1e-1` | 20 | 0.2104 | 0.1900 | Started, stopped at 475/1000 |
| `dino_channel_speed_constancy` | `speed1e-1` | 17 | 0.0939 | 0.0742 | Did not reach planner |
| `dino_channel_cos_plus_speed` | `cos1e-1+speed1e-1` | 20 | 0.1856 | 0.1680 | Started, stopped at 805/1000 |

## Methodology

The Medium dataset was generated on Modal as 50 random-policy episodes with 100 frames per episode. Each model trained for 20 epochs and was evaluated with gradient-descent planning using `n_evals=50` and `goal_H=25`. Planner evidence used corrected fixed planner runs after the original full-run planner command hit stale Hydra overrides.

The speed ablation added a speed constancy term to the existing temporal straightening code. The original cosine straightening minimizes directional curvature by aligning consecutive velocity vectors in latent space. The added term penalizes variation in consecutive latent step magnitudes, addressing the gap that cosine distance normalizes away speed.

The initial ablation was intentionally conservative but has a limitation: DINO patch and CLS encoders were frozen. Therefore, speed losses can change predictor/decoder training losses, but cannot change the frozen encoder latent geometry measured by `model.encode_obs(obs)["visual"]`. To address that, I added and launched adapter variants using `encoder=dino_channel`, where DINO's backbone is frozen but the channel projector and aggregation MLP are trainable and lie on the measured latent path.

## Findings

1. Medium Ours vs DINO: in this reproduction, authors-style patch cosine straightening and frozen DINO patch both reached 0.14 success. DINO CLS reached 0.22 success and lower mean state distance.

2. Frozen speed ablations support the critique but do not prove a fix. The speed terms changed training losses but did not change planner success or measured latent diagnostics because the evaluated DINO latent geometry was frozen.

3. Image evidence aligns with the frozen-encoder finding. Patch speed and patch cosine+speed contact sheets are byte-identical for validation and rollout images. CLS images differ from patch images. This matches the experiment result: same encoder family and frozen latent path gives identical visual evidence; changing encoder family changes images.

4. Trainable adapter runs are the right next experiment. They address the caveat because the measured latent path includes trainable projector parameters. However, final planner and latent results are incomplete due to the Modal spend limit.

5. Wall was not reproducible from the checkout as-is. The Wall wrapper imports `env.wall.data.wall`, `env.wall.data.wall_utils`, and `env.wall.data.single`, but those files were missing in both the local checkout and upstream. The Modal volume also did not contain `wall_single`. I added the missing Wall support modules and a `generate_wall_dataset.py` generator matching `datasets/wall_dset.py`.

## Wall Code Work

Added:

- `env/wall/data/wall.py`: `WallDatasetConfig` dataclass used by the wrapper.
- `env/wall/data/wall_utils.py`: fixed-layout and generalized wall/door layout enumeration.
- `env/wall/data/single.py`: compatibility import placeholder.
- `generate_wall_dataset.py`: creates `states.pth`, `actions.pth`, `door_locations.pth`, `wall_locations.pth`, `seq_lengths.pth`, and `obses/episode_*.pth`.
- `datasets/wall_dset.py`: handles saved HWC or CHW image tensors robustly.
- `modal_medium_runner.py`: adds `--environment medium|wall`, Wall dataset generation, adapter ablation variants, and no-worker dataloader compatibility.

Blocked command:

```bash
modal run modal_medium_runner.py --action run --environment wall --run-id wall-smoke-20260702-01 --epochs 1 --n-episodes 3 --episode-length 40 --batch-size 2 --num-workers 0 --plan-n-evals 1 --plan-goal-h 10 --variant-name authors_dino_patch_straightened
```

Modal returned:

```text
App creation failed: workspace billing cycle spend limit reached
```

## Evidence Index

- Medium completed summary: `modal_evidence/medium-full-20260702-01/combined_results.json`
- Medium latent diagnostics: `modal_evidence/medium-full-20260702-01/latent_analysis.json`
- Frozen speed ablation summary: `modal_evidence/medium-ablations-20260702-01/combined_medium_results.json`
- Image evidence summary: `modal_evidence/medium-ablations-20260702-01/image_evidence_summary.json`
- Adapter partial evidence: `modal_evidence/medium-adapter-ablations-20260702-01/partial_adapter_results.json`
- Contact sheets:
  - `modal_evidence/medium-full-20260702-01/epoch20_validation_contact_sheet.png`
  - `modal_evidence/medium-full-20260702-01/epoch20_rollout_train0_contact_sheet.png`
  - `modal_evidence/medium-ablations-20260702-01/ablation_epoch20_validation_contact_sheet.png`
  - `modal_evidence/medium-ablations-20260702-01/ablation_epoch20_train0_rollout_contact_sheet.png`

Large checkpoint and tar archives remain local and on the Modal volume, but are not suitable for normal GitHub PR upload because several files are 376 MB to 1.3 GB and the local evidence tree is 33 GB.

## Next Steps

When Modal spend is available again:

1. Finish the four `dino_channel` adapter planners and run latent diagnostics.
2. Run Wall smoke, then Wall Ours and DINO:

```bash
modal run modal_medium_runner.py --action run --environment wall --run-id wall-full-20260703-01 --epochs 20 --n-episodes 50 --episode-length 100 --batch-size 32 --num-workers 0 --plan-n-evals 50 --plan-goal-h 25 --variant-name authors_dino_patch_straightened
modal run modal_medium_runner.py --action run --environment wall --run-id wall-dino-20260703-01 --epochs 20 --n-episodes 50 --episode-length 100 --batch-size 32 --num-workers 0 --plan-n-evals 50 --plan-goal-h 25 --variant-name dino_patch_no_straightening
```

3. Package Wall artifacts with `--action package-artifacts` and update this report with final Wall metrics.

