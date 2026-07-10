# Declarative custom runs

Each `*.yaml` file in this directory is a constrained request, not a Hydra
manifest and not a command. The trusted compiler accepts exactly these fields:

```yaml
schema_version: 1
name: normacc_probe # must match normacc_probe.yaml
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

Allowed variants are `projector_only`, `paper_curvature`, `curvature`,
`ratio_speed`, and `normalized_acceleration`. Evaluation is either `proxy` or
`proxy_and_plan`. The compiler rejects unknown fields, so a request cannot add
Hydra overrides, commands, modules, images, AWS resources, attempts, or input
locations.

Bounds are 10–800 rollouts, 1–20 epochs, 2–50 goals, 1–3 unique seed triples,
0.25–12 hours per run, and $0.25–$2.50 per run. The declared maximum across all
runs and both fixed Batch attempts must be at most $5.

Validate before pushing:

```bash
python -m infra.experiments.custom_manifest validate \
  infra/experiments/manifests/custom/normacc_probe.yaml
```

After the spec is reviewed and is the current `infra` tip, the filename stem
selects it. The final tag component is an audit nonce:

```bash
git tag train-run-normacc_probe-utsav01
git push origin train-run-normacc_probe-utsav01
```
