# UMaze probe walkthrough

## Full paper-protocol ON/OFF notebook

`umaze_layerwise_motion_probe_paper_lr.ipynb` is the current full comparison.
It clones `codex/umaze-physics-probes`, reconstructs both frozen checkpoints
from the verified parts under
`baseline_artifacts/checkpoints/umaze_paper_protocol_on_off`, and downloads the
2,000-trajectory public UMaze dataset directly from OSF. It does not mount
Google Drive, request private uploads, or require saved credentials.

The OFF and ON checkpoint SHA-256 values, reconstructed provenance sidecars,
and restoration script are checked into the same branch. A GPU Colab runtime
can therefore run the entire notebook from the repository alone.

## Colab: credential-free walkthrough

Open `umaze_layerwise_motion_probe_walkthrough.ipynb` through GitHub and choose
**Run all**.  Its first code cell clones `codex/umaze-physics-probes` and
downloads the checksummed `umaze-probe-colab-v1` activation cache from the
repository's public GitHub release.  Collaborators do not need an R2 account,
the 58 GB raw trajectory directory, or the 254 MB checkpoint for this default
path.  The notebook still refits every ridge probe, reruns both held-out splits
and controls, and regenerates all tables and figures.

The public cache stores frozen R0 intermediate activations, physical states,
actions, and deterministic window identifiers.  It does not contain raw
trajectory images or credentials.

To recompute the activations instead of using the public cache, set
`UMAZE_FORCE_RECOMPUTE=1`, `UMAZE_CHECKPOINT`, and `UMAZE_DATA_DIR` before the
setup cell.  Full recomputation is intentionally an advanced path because the
source trajectory directory is too large for a normal Colab session.

`umaze_layerwise_motion_probe_walkthrough.ipynb` is the executable,
explanatory version of the UMaze layerwise physics probes. It compares where
position, velocity, and acceleration become linearly readable in DINO and the
action/proprioception-conditioned predictor.

## What it adds beyond the original probes

- DINO frame, first-difference, and second-difference inputs are reported
  separately.
- The default split holds out complete episodes rather than randomly mixing
  trajectory windows.
- A spatially blocked split checks generalization to an unseen maze region.
- Shuffled-label, position-only, and position-residualized controls test whether
  apparent motion decoding is only a location shortcut.
- Cartesian and polar targets are shown layer by layer.
- A conservative, explicit rule creates an emergence-layer summary.

## Required inputs for full recomputation

1. An epoch checkpoint containing `encoder`, `predictor`, `proprio_encoder`, and
   `action_encoder` modules.
2. The UMaze trajectory dataset with images, simulator states, actions, and
   episode boundaries.
3. The repository training environment with PyTorch, NumPy, and Matplotlib.

Set paths before opening Jupyter if the defaults do not match the machine:

```bash
export UMAZE_CHECKPOINT=/path/to/model_20.pth
export UMAZE_DATA_DIR=/path/to/point_maze
export UMAZE_DEVICE=cuda:0
export UMAZE_PROBE_OUTPUT=/path/to/analysis/umaze_probe_walkthrough
```

On the A100 machines used for the original study, the usual inputs are:

```bash
export UMAZE_CHECKPOINT="$HOME/temporal-straightening/baseline_artifacts/checkpoints/umaze_q1_retrain/r0_direction_only/checkpoints/model_20.pth"
export UMAZE_DATA_DIR="$HOME/data/point_maze"
```

If those paths are absent, restore the matching checkpoint and dataset from the
configured `r2` AWS profile before execution. Do not put R2 credentials in the
notebook; keep them in the normal AWS profile.

## Execution

The first successful run extracts pooled intermediate activations on one GPU
and writes `activation_cache_pooled.npz`. Later executions reuse that cache and
run the probes on CPU. Delete the cache whenever the checkpoint, dataset,
frame spacing, or sampled windows change.

If Jupyter's default kernel is not the training environment, register it once
and select **UMaze Probe (ts)** from the notebook's kernel menu:

```bash
/path/to/training/venv/bin/python -m ipykernel install --user \
  --name umaze-probe --display-name "UMaze Probe (ts)"
```

The notebook writes CSV tables, JSON summaries, and PNG figures under
`baseline_artifacts/analysis/umaze_probe_walkthrough` by default.

Regenerate the checked-in notebook after editing its source cells with:

```bash
python scripts/build_umaze_probe_notebook.py
```

The notebook is intentionally generated from a Python script so cell source is
reviewable in ordinary Git diffs.
