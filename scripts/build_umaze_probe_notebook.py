#!/usr/bin/env python3
"""Generate the checked-in UMaze layerwise motion-probe walkthrough notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# Where does UMaze physics become readable?

This notebook reproduces and strengthens the repository's UMaze linear-probe methodology. It asks where **position**, **velocity**, and **acceleration** become linearly readable across DINO and predictor layers.

The key distinction is temporal context:

- DINO encodes one image at a time. Position is probed from $h_t$, while the primary DINO velocity and acceleration probes use $\Delta h_t$ and $\Delta^2h_t$.
- The predictor receives visual tokens together with action/proprioception and causal history, so its per-slot activation may legitimately contain motion information.

“Readable” means a frozen representation supports a held-out ridge-linear decoder. It does **not** prove the planner causally uses that variable."""
    ),
    markdown(
        """## Experimental design

For a layer representation $h_t^\ell$:

$$p_t=(x_t,y_t),\qquad v_t=\\frac{p_{t+1}-p_t}{\Delta t},\qquad a_t=\\frac{v_{t+1}-v_t}{\Delta t}$$

Primary feature/target pairs:

| Model family | Position | Velocity | Acceleration |
|---|---|---|---|
| DINO | $h_t^\ell$ | $h_{t+1}^\ell-h_t^\ell$ | $h_{t+1}^\ell-2h_t^\ell+h_{t-1}^\ell$ |
| Predictor | contextual $h_t^\ell$ | contextual $h_t^\ell$ | contextual $h_t^\ell$ |

Every probe standardizes features using training statistics and fits ridge regression with $\lambda=10$. We compare a random episode-held-out split and a spatially blocked split, and include shuffled-label, position-only, and position-residualized controls."""
    ),
    code(
        """from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "scripts").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from scripts.umaze_probe_walkthrough import (
    align_representation, bootstrap_metric_ci, build_motion_targets,
    direction_scores, episode_group_split, fit_probe, load_activation_cache,
    mask_slow_directions, readability_onset, regression_scores, residualize_against_position,
    save_activation_cache, shuffled_label_score, spatial_holdout_split,
)

sns.set_theme(style="whitegrid", context="notebook")
SEED = 0
RIDGE = 10.0
MAX_WINDOWS = 512
NUM_FRAMES = 4
FRAME_SKIP = 5
STEP_DT = 1.0  # set to the environment seconds-per-step if physical units are needed
BATCH_SIZE = 8

CHECKPOINT = Path(os.environ.get("UMAZE_CHECKPOINT", ROOT / "baseline_artifacts/checkpoints/umaze_q1_retrain/r0_direction_only/checkpoints/model_20.pth"))
DATA_DIR = Path(os.environ.get("UMAZE_DATA_DIR", Path.home() / "data/point_maze"))
OUTPUT_DIR = Path(os.environ.get("UMAZE_PROBE_OUTPUT", ROOT / "baseline_artifacts/analysis/umaze_probe_walkthrough"))
CACHE = Path(os.environ.get("UMAZE_ACTIVATION_CACHE", OUTPUT_DIR / "activation_cache_pooled.npz"))
DEVICE = os.environ.get("UMAZE_DEVICE", "cuda:0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print({"checkpoint": str(CHECKPOINT), "data": str(DATA_DIR), "cache": str(CACHE), "device": DEVICE})"""
    ),
    markdown(
        """## 1. Collect or load intermediate activations

The cache contains states, actions, sampled `(episode, start)` windows, and pooled representations. Individual patches are intentionally excluded from the first pass because retaining every patch at every DINO layer can consume several gigabytes. Delete the cache to force re-extraction after changing the checkpoint, dataset, or window configuration."""
    ),
    code(
        """if CACHE.exists():
    representations, states, actions, choices, cache_metadata = load_activation_cache(CACHE)
    print(f"Loaded {len(representations)} representations from {CACHE}")
else:
    import torch
    from datasets.img_transforms import default_transform
    from datasets.point_maze_dset import PointMazeDataset
    from scripts.probe_umaze_layers import collect_activations, load_checkpoint, sample_windows

    if not CHECKPOINT.exists() or not DATA_DIR.exists():
        raise FileNotFoundError(
            "Set UMAZE_CHECKPOINT and UMAZE_DATA_DIR, or copy the checkpoint/data from R2, "
            "before running activation extraction."
        )
    device = torch.device(DEVICE)
    modules = load_checkpoint(CHECKPOINT, device)
    use_frame_files = (DATA_DIR / "obses" / "episode_000_frame_000.pth").exists()
    dataset = PointMazeDataset(
        data_path=str(DATA_DIR), transform=default_transform(224),
        normalize_action=True, use_frame_files=use_frame_files,
    )
    choices = sample_windows(dataset, MAX_WINDOWS, FRAME_SKIP, NUM_FRAMES, SEED)
    representations, states, actions = collect_activations(
        modules, dataset, choices, BATCH_SIZE, FRAME_SKIP, NUM_FRAMES, device,
        include_kinds={"cls", "pooled_patches", "projected_aggregate", "pooled_visual"},
    )
    cache_metadata = {
        "checkpoint": str(CHECKPOINT), "data_dir": str(DATA_DIR), "seed": SEED,
        "max_windows": MAX_WINDOWS, "num_frames": NUM_FRAMES,
        "frameskip": FRAME_SKIP, "ridge": RIDGE,
    }
    save_activation_cache(CACHE, representations, states, actions, choices, cache_metadata)
    print(f"Saved pooled activation cache to {CACHE}")

inventory = pd.DataFrame([
    {"representation": name, "shape": str(value.shape), "size_mb": value.nbytes / 2**20}
    for name, value in sorted(representations.items())
])
display(inventory)
print("Total cached representation memory (MB):", inventory.size_mb.sum())"""
    ),
    markdown(
        """## 2. Construct physical targets and inspect shortcut risk

Position comes from `state[..., :2]`. Instantaneous velocity uses `state[..., 2:4]` when present. Labels paired with temporal DINO differences always use the matching finite displacement. Acceleration is a finite difference of velocity.

Before fitting a model probe, inspect whether speed and acceleration already correlate with maze location. Strong spatial structure is exactly the shortcut a single-frame DINO probe could exploit."""
    ),
    code(
        """targets = build_motion_targets(states, FRAME_SKIP, STEP_DT)
position = targets["position"].reshape(-1, 2)
speed = targets["speed"].reshape(-1)
acceleration_magnitude = targets["acceleration_magnitude"].reshape(-1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
axes[0].scatter(position[:, 0], position[:, 1], c=speed, s=8, cmap="viridis")
axes[0].set(title="UMaze coverage colored by speed", xlabel="x", ylabel="y", aspect="equal")
sns.histplot(speed, bins=40, ax=axes[1]); axes[1].set_title("Speed distribution")
sns.histplot(acceleration_magnitude[np.isfinite(acceleration_magnitude)], bins=40, ax=axes[2])
axes[2].set_title("Acceleration-magnitude distribution")
fig.savefig(OUTPUT_DIR / "dataset_motion_overview.png", dpi=180)
plt.show()

spatial_table = pd.DataFrame({"x": position[:, 0], "y": position[:, 1], "speed": speed})
spatial_table["x_bin"] = pd.qcut(spatial_table.x, 12, duplicates="drop")
spatial_table["y_bin"] = pd.qcut(spatial_table.y, 12, duplicates="drop")
speed_map = spatial_table.pivot_table(index="y_bin", columns="x_bin", values="speed", observed=True)
plt.figure(figsize=(8, 6)); sns.heatmap(speed_map, cmap="mako")
plt.title("Mean speed by spatial bin (shortcut diagnostic)")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "speed_by_position.png", dpi=180); plt.show()"""
    ),
    markdown(
        """## 3. Leakage-resistant evaluation splits

- **Episode-held-out:** complete trajectories are assigned to train or test. This prevents nearby or overlapping windows from the same episode crossing the boundary.
- **Spatial holdout:** the upper 20% of window-anchor Y positions is test-only, with a buffer band removed from training. Change the axis/direction to rotate through every arm and corner.

The split is made at the window level and then shared by every representation and layer."""
    ),
    code(
        """episode_train, episode_test = episode_group_split(choices, test_fraction=0.2, seed=SEED)
anchor_position = targets["position"][:, 0]
spatial_train, spatial_test, spatial_config = spatial_holdout_split(
    anchor_position, axis=1, quantile=0.8, high=True, buffer_fraction=0.05
)
splits = {
    "episode_holdout": (episode_train, episode_test),
    "spatial_holdout": (spatial_train, spatial_test),
}
print({name: (len(train), len(test)) for name, (train, test) in splits.items()})
print("Spatial split:", spatial_config)

plt.figure(figsize=(6, 5))
plt.scatter(anchor_position[spatial_train, 0], anchor_position[spatial_train, 1], s=18, label="train")
plt.scatter(anchor_position[spatial_test, 0], anchor_position[spatial_test, 1], s=18, label="held-out region")
plt.legend(); plt.gca().set_aspect("equal"); plt.title("Spatially blocked split")
plt.tight_layout(); plt.savefig(OUTPUT_DIR / "spatial_split.png", dpi=180); plt.show()"""
    ),
    markdown(
        """## 4. Fit every layer with controls

For each representation we record:

- held-out $R^2$, RMSE, and a bootstrap interval;
- a shuffled-label null;
- a position-only baseline predicting the same target from XY;
- residual-motion readability after subtracting the component predictable from XY.

The residual score is especially important for raw per-frame motion probes."""
    ),
    code(
        """def evaluate_representation(name, rep, variable, mode, split_name, train_idx, test_idx):
    features, labels, position_context = align_representation(rep, targets, variable, mode)
    truth, prediction, _ = fit_probe(features, labels, train_idx, test_idx, RIDGE)
    scores = regression_scores(truth, prediction)
    ci = bootstrap_metric_ci(truth, prediction, "r2", repeats=300, seed=SEED)
    shuffled = shuffled_label_score(
        features, labels, train_idx, test_idx, RIDGE, repeats=20, seed=SEED
    )
    pos_truth, pos_prediction, _ = fit_probe(
        position_context, labels, train_idx, test_idx, RIDGE
    )
    residual_labels = residualize_against_position(labels, position_context, train_idx, RIDGE)
    residual_truth, residual_prediction, _ = fit_probe(
        features, residual_labels, train_idx, test_idx, RIDGE
    )
    family, layer, kind = name.split("/", 2)
    return {
        "representation": name, "family": family, "layer": int(layer), "kind": kind,
        "variable": variable, "mode": mode, "split": split_name,
        **scores, "ci_low": ci[0], "ci_high": ci[1],
        "shuffled_q95": float(np.quantile(shuffled, 0.95)),
        "position_only_r2": regression_scores(pos_truth, pos_prediction)["r2"],
        "position_residual_r2": regression_scores(residual_truth, residual_prediction)["r2"],
    }

rows = []
for split_name, (train_idx, test_idx) in splits.items():
    for name, rep in sorted(representations.items()):
        family = name.split("/", 1)[0]
        specs = [("position", "frame")]
        if family == "dino":
            specs += [("velocity", "frame"), ("velocity", "delta")]
            if rep.shape[1] >= 3:
                specs += [("acceleration", "frame"), ("acceleration", "second_delta")]
        else:
            specs += [("velocity", "frame"), ("acceleration", "frame")]
        for variable, mode in specs:
            rows.append(evaluate_representation(
                name, rep, variable, mode, split_name, train_idx, test_idx
            ))

metrics = pd.DataFrame(rows)
metrics.to_csv(OUTPUT_DIR / "layerwise_cartesian_metrics.csv", index=False)
display(metrics.head())"""
    ),
    markdown(
        """## 5. Where each Cartesian variable becomes readable

The primary DINO curves use a raw frame for position, a first temporal difference for velocity, and a second temporal difference for acceleration. Predictor curves use contextual per-slot features. Shaded regions are bootstrap intervals over held-out rows."""
    ),
    code(
        """def primary_mode(family, variable):
    if family == "predictor" or variable == "position":
        return "frame"
    return {"velocity": "delta", "acceleration": "second_delta"}[variable]

def plot_layer_curves(frame, split="episode_holdout", filename="readability_by_layer.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    for ax, variable in zip(axes, ["position", "velocity", "acceleration"]):
        for (family, kind), group in frame[(frame.split == split) & (frame.variable == variable)].groupby(["family", "kind"]):
            group = group[group["mode"] == primary_mode(family, variable)].sort_values("layer")
            if group.empty:
                continue
            label = f"{family}: {kind}"
            ax.plot(group.layer, group.r2, marker="o", label=label)
            ax.fill_between(group.layer, group.ci_low, group.ci_high, alpha=0.12)
        ax.axhline(0, color="black", lw=1)
        ax.set(title=f"{variable.title()} readability", xlabel="layer", ylabel="held-out R²")
        ax.legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.show()

plot_layer_curves(metrics)
plot_layer_curves(metrics, split="spatial_holdout", filename="readability_by_layer_spatial_holdout.png")"""
    ),
    markdown(
        """## 6. Is DINO motion static or genuinely temporal?

This is the core shortcut check. A high raw-frame score that disappears on the spatial holdout can be explained by location. A temporal-difference score that beats the position-only baseline and survives in unseen regions is stronger evidence that representation change tracks motion."""
    ),
    code(
        """dino_motion = metrics[
    (metrics.family == "dino") & metrics.variable.isin(["velocity", "acceleration"])
].copy()
g = sns.relplot(
    data=dino_motion, x="layer", y="r2", hue="mode", style="kind",
    col="variable", row="split", kind="line", marker="o",
    facet_kws={"sharey": False}, height=3.4, aspect=1.4,
)
g.set_axis_labels("DINO layer", "held-out R²")
g.fig.suptitle("Raw per-frame versus temporal DINO probes", y=1.02)
g.savefig(OUTPUT_DIR / "static_vs_temporal_dino.png", dpi=180)
plt.show()

control_view = dino_motion[dino_motion["mode"].isin(["delta", "second_delta"])].copy()
display(control_view.sort_values("r2", ascending=False)[[
    "split", "variable", "representation", "r2", "position_only_r2",
    "position_residual_r2", "shuffled_q95"
]].head(20))"""
    ),
    markdown(
        """## 7. Cartesian versus polar motion

Velocity is also probed as speed plus heading $(\cos\\theta,\sin\\theta)$. Acceleration is decomposed the same way. Direction metrics exclude the slowest 10% of samples, where angle is poorly defined."""
    ),
    code(
        """polar_rows = []
train_idx, test_idx = splits["episode_holdout"]
for name, rep in sorted(representations.items()):
    family, layer, kind = name.split("/", 2)
    velocity_mode = "delta" if family == "dino" else "frame"
    acceleration_mode = "second_delta" if family == "dino" else "frame"
    for variable, mode in [
        ("speed", velocity_mode), ("heading", velocity_mode),
        ("acceleration_magnitude", acceleration_mode),
        ("acceleration_direction", acceleration_mode),
    ]:
        if mode == "second_delta" and rep.shape[1] < 3:
            continue
        features, labels, _ = align_representation(rep, targets, variable, mode)
        if variable in ("heading", "acceleration_direction"):
            magnitude_variable = "speed" if variable == "heading" else "acceleration_magnitude"
            _, magnitude, _ = align_representation(rep, targets, magnitude_variable, mode)
            labels, cutoff = mask_slow_directions(labels, magnitude, train_idx, quantile=0.1)
        truth, prediction, _ = fit_probe(features, labels, train_idx, test_idx, RIDGE)
        score = direction_scores(truth, prediction) if "direction" in variable or variable == "heading" else regression_scores(truth, prediction)
        polar_rows.append({
            "representation": name, "family": family, "layer": int(layer), "kind": kind,
            "variable": variable, "mode": mode, **score,
        })

polar_metrics = pd.DataFrame(polar_rows)
polar_metrics.to_csv(OUTPUT_DIR / "layerwise_polar_metrics.csv", index=False)
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
for ax, (variable, score_key) in zip(axes.flat, [
    ("speed", "r2"), ("heading", "cosine"),
    ("acceleration_magnitude", "r2"), ("acceleration_direction", "cosine"),
]):
    subset = polar_metrics[polar_metrics.variable == variable]
    for (family, kind), group in subset.groupby(["family", "kind"]):
        group = group.sort_values("layer")
        ax.plot(group.layer, group[score_key], marker="o", label=f"{family}: {kind}")
    ax.axhline(0, color="black", lw=1)
    ax.set(title=variable.replace("_", " ").title(), xlabel="layer", ylabel=score_key)
    ax.legend(fontsize=7)
fig.savefig(OUTPUT_DIR / "cartesian_vs_polar_by_layer.png", dpi=180)
plt.show()"""
    ),
    markdown(
        """## 8. Emergence table

A conservative onset is the first of two consecutive layers whose $R^2$ exceeds the shuffled-label 95th percentile and reaches at least half of that representation family's peak score. This definition should be reported alongside the full curves rather than treated as a uniquely correct boundary."""
    ),
    code(
        """onsets = []
primary = metrics[metrics.split == "episode_holdout"].copy()
primary = primary[
    primary.apply(lambda row: row["mode"] == primary_mode(row["family"], row["variable"]), axis=1)
]
for (family, kind, variable), group in primary.groupby(["family", "kind", "variable"]):
    onsets.append({
        "family": family, "kind": kind, "variable": variable,
        "onset_layer": readability_onset(
            group.to_dict("records"), "r2", "shuffled_q95", consecutive=2, fraction_of_peak=0.5
        ),
        "peak_r2": group.r2.max(), "peak_layer": int(group.loc[group.r2.idxmax(), "layer"]),
    })
onset_table = pd.DataFrame(onsets)
onset_table.to_csv(OUTPUT_DIR / "readability_onsets.csv", index=False)
display(onset_table.sort_values(["variable", "family", "kind"]))"""
    ),
    markdown(
        """## 9. Interpretation checklist

Use these rules when writing the result:

1. **Position:** a high per-frame DINO score is expected and validates the probe pipeline.
2. **Velocity:** prioritize DINO $\Delta h$ and contextual predictor scores. Treat raw-frame DINO velocity as a shortcut diagnostic.
3. **Acceleration:** prioritize DINO $\Delta^2h$ and contextual predictor scores; compare its onset with velocity.
4. **Location generalization:** report whether motion performance survives the spatial holdout and beats the position-only baseline.
5. **Causality:** never infer causal use from readability alone. Interventions or steering along probe directions are a separate experiment.

The strongest defensible claim has the form: “Variable X becomes linearly readable at layer L, beats shuffled and position-only controls, and generalizes to a held-out maze region.”"""
    ),
    code(
        """summary = {
    "config": {
        **cache_metadata, "ridge": RIDGE, "step_dt": STEP_DT,
        "episode_split": {"train_windows": len(episode_train), "test_windows": len(episode_test)},
        "spatial_split": spatial_config,
    },
    "onsets": onset_table.where(pd.notna(onset_table), None).to_dict("records"),
    "best_cartesian_rows": metrics.sort_values("r2", ascending=False).head(20).to_dict("records"),
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("Wrote:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)"""
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "notebooks" / "umaze_layerwise_motion_probe_walkthrough.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    print(output)


if __name__ == "__main__":
    main()
