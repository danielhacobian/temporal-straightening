# Experiment manifests

This directory turns the paper anchor and screening funnel into deterministic,
provider-neutral run specifications. AWS Batch only supplies a container, GPU,
array index, environment variables, and `/scratch`; no module imports Modal or
calls a cloud control-plane API.

## Build the fixed UMaze goal bundle

Generate the shared bundle once from the released 2,000-trajectory UMaze
ZIP. The default path loads no model, needs no GPU or MuJoCo installation, and
does not expand the roughly 30 GB observation archive. It validates the whole
ZIP directory, reads the three core tensors, and deserializes only the 100
observation members assigned to planner or proxy evaluation.

```bash
python -m infra.experiments.build_goal_bundle \
  --dataset-zip point_maze_umaze.zip \
  --dataset-sha256 64_HEX_CHARACTER_RELEASE_DIGEST \
  --dataset-version-id S3_VERSION_ID \
  --output umaze_fixed_v1.pkl

sha256sum -c umaze_fixed_v1.pkl.sha256
```

The builder exactly matches the repository's trajectory split (90/10,
seed 42) and reserves disjoint validation episodes: 50 for 25-step planner
segments and 50 for fixed proxy trajectories. Planner endpoints are the
released raw HWC frames aligned to their archived states/actions. Actions use
the same valid-step normalization as training and are stored as five groups of
five low-level actions.

Proxy frames are selected at evenly spaced times before any label is computed;
the builder then stores BFS shortest-path distance to the fixed final state.
Candidates with fewer than three distinct sampled BFS levels are excluded
because within-trajectory rank correlation is degenerate there. That fixed
eligibility rule never observes model latents or changes selected frames; the
bundle records it explicitly. This preserves ties and backtracking and never
substitutes remaining time for path distance. Provenance includes the release ZIP digest/version, core and
selected-member digests, source episode/frame IDs, split seeds, and an explicit
planner/proxy disjointness assertion. The pickle and its
`umaze_fixed_v1.pkl.sha256` sidecar are published atomically.

The older MuJoCo replay path remains available for a diagnostic comparison:

```bash
python -m infra.experiments.build_goal_bundle \
  --dataset-dir data/point_maze \
  --output umaze_replayed_diagnostic.pkl
```

## Required S3 variables

The manifests defer account-specific locations to the Batch job environment:

```bash
export TS_UMAZE_DATASET_S3_URI=s3://BUCKET/datasets/point_maze_umaze.zip
export TS_UMAZE_DATASET_SHA256=64_HEX_CHARACTERS
export TS_UMAZE_DATASET_VERSION_ID=S3_VERSION_ID
export TS_UMAZE_GOALS_S3_URI=s3://BUCKET/goals/umaze_fixed_v1.pkl
export TS_UMAZE_GOALS_SHA256=64_HEX_CHARACTERS
export TS_UMAZE_GOALS_VERSION_ID=S3_VERSION_ID
export TS_ARTIFACT_PREFIX=s3://BUCKET/artifacts
```

The released dataset archive must contain `point_maze/`. The goal pickle uses
the existing `plan.py` contract (`obs_0`, `obs_g`, `state_0`, `state_g`,
`gt_actions`, and `goal_H`) plus `proxy_trajectories`. Each proxy entry is one
held-out trajectory/goal group with `observations`, one `goal_observation`, and
either authoritative `shortest_path_steps` or `states` plus `goal_state`.
Spearman is computed within each trajectory/goal and only then summarized
across groups. Independent start/goal pairs are rejected because they cannot
measure monotonicity along a trajectory. This remains a candidate signal until
it predicts new planner results prospectively.

## Validate, plan, and run

Static validation does not need AWS credentials or resolved S3 variables:

```bash
python -m infra.experiments.runner validate infra/experiments/manifests/*.yaml
```

Planning resolves the variables and emits the complete Batch array in stable
index order. The optional submission budget is a preflight guard, not an AWS
billing alarm:

```bash
python -m infra.experiments.runner plan \
  infra/experiments/manifests/umaze_exact_anchor.yaml \
  --profile minimal --budget-usd 20 --output plan.json
```

The released-dataset paper-recipe anchor is 3 variants x 3 matched seeds.
`--profile minimal` keeps the paper's projector-only versus
projector+curvature comparison (6 runs). All three training seeds use the
identical canonical released dataset order, and every variant shares the same
fixed 50-goal bundle. The plan reports its maximum aggregate envelope; it intentionally refuses a
`--budget-usd` lower than that amount rather than pretending the anchor fits.

An AWS Batch array worker reads `AWS_BATCH_JOB_ARRAY_INDEX`:

```bash
python -m infra.experiments.runner run \
  infra/experiments/manifests/umaze_exact_anchor.yaml --profile minimal
```

Set `EXPERIMENT_HOURLY_USD` when the infrastructure knows the applicable
instance rate. The runner rejects rates above the manifest cap and derives a
second timeout from `max_usd / hourly_rate`; AWS Batch should also set its job
timeout to `max_hours`. Without the variable, the runner conservatively uses
the manifest's hourly cap.

Each worker first restores an existing S3 run root, then downloads and
checksum-verifies the dataset and fixed goals, materializes a seeded rollout
subset in `/scratch`, injects separate data, training, and planner seeds, and
uploads the entire run root (metadata, results, epoch checkpoints, and logs) to
S3 even after interruption or most failures. Tiny screening jobs postpone
decoder loss; the released-dataset anchor keeps the paper-recipe decoder path
active from epoch 1.

## Funnel boundaries

- `smoke.yaml`: one 10-rollout, 1-epoch projector+curvature job that also runs
  the held-out proxy and a 2-goal/2-optimizer-step planner check.
- `umaze_exact_anchor.yaml`: released-dataset paper-v1 recipe anchor, with full
  and minimal profiles. It varies training seed on one canonical dataset; it
  does not reproduce Table 1's three independently sampled datasets.
- `screening_funnel.yaml`: 100 rollouts, epoch 10, three seeds, proxy only.
- `finalists.yaml`: projector-only, paper curvature, and the current normalized-
  acceleration candidate under paired goals and seeds.
- `scaling_trend.yaml`: two variants at 50, 200, and 800 rollouts; six trend
  runs, not a definitive paper-scale result.

The ℛ4 raw-acceleration control is represented but disabled in the screening
manifest because the current model parser does not implement that loss. This
prevents an unrecognized token from being mislabeled as a real control run.

## Declarative contributor runs

Contributors can request a bounded custom run without placing a command in a
tag. Add a strict request under `manifests/custom/`; for example,
`manifests/custom/normacc_probe.yaml`:

```yaml
schema_version: 1
name: normacc_probe
variant: normalized_acceleration
rollouts: 50
epochs: 10
evaluation: proxy
goal_count: 50
seeds:
  - data_seed: 10
    train_seed: 20
    planner_seed: 100
limits:
  max_hours: 4
  max_usd: 2
```

Validate and push the spec to `infra`, then tag that exact current tip:

```bash
python -m infra.experiments.custom_manifest validate \
  infra/experiments/manifests/custom/normacc_probe.yaml
git tag train-run-normacc_probe-utsav01
git push origin train-run-normacc_probe-utsav01
```

The filename stem and `name` must match `[a-z0-9][a-z0-9_]{0,31}`. The final
tag component is an audit nonce, not an argument. The trusted compiler rejects
unknown fields and maps only five approved variants and two evaluation modes
to literal Hydra settings. Input URIs, object versions, image, entrypoint,
GPU count, two Batch attempts, hourly cap, and artifact prefix are not fields
contributors can set.

Custom requests allow 10–800 rollouts, 1–20 epochs, 2–50 goals, at most three
seed triples, no more than 12 hours or $2.50 per run, and at most $5 across the
entire array and both attempts. See
[`manifests/custom/README.md`](manifests/custom/README.md) for the full closed
schema.
