#!/usr/bin/env python3
"""Layerwise physical-variable probes for a trained UMaze visual world model.

The script uses held-out trajectory windows and reports how linearly available
physical speed, direction, collision/stall, A* distance, and action magnitude
are in DINO and predictor activations.  It also measures whether the A* probe
direction overlaps the latent subspace that actions can control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

from datasets.img_transforms import default_transform
from datasets.point_maze_dset import PointMazeDataset
from scripts.evaluate_umaze_latent_geodesic import (
    astar_step_counts,
    make_occupancy_grid,
    nearest_valid_node,
    resolve_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ridge", type=float, default=10.0)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    # Register torch-hub DINO classes before unpickling old checkpoints.
    from models.dino import DinoV2Encoder

    _ = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    try:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolve_checkpoint(path), map_location="cpu")
    modules = {}
    for name in ("encoder", "predictor", "proprio_encoder", "action_encoder"):
        if name not in payload:
            raise KeyError(f"checkpoint is missing {name!r}")
        modules[name] = payload[name].to(device).eval()
    return modules


def sample_windows(dataset: PointMazeDataset, count: int, frameskip: int, nframes: int, seed: int):
    rng = random.Random(seed)
    choices = []
    for episode, length in enumerate(dataset.seq_lengths.tolist()):
        max_start = int(length) - 1 - frameskip * (nframes - 1)
        if max_start >= 0:
            choices.extend((episode, start) for start in range(max_start + 1))
    rng.shuffle(choices)
    return choices[: min(count, len(choices))]


def load_batch(dataset, choices, frameskip, nframes):
    visuals, proprios, actions, states = [], [], [], []
    for episode, start in choices:
        indices = [start + frameskip * offset for offset in range(nframes)]
        obs, act, state, _ = dataset.get_frames(episode, indices)
        visuals.append(obs["visual"])
        proprios.append(obs["proprio"])
        actions.append(act)
        states.append(state)
    return (
        torch.stack(visuals),
        torch.stack(proprios),
        torch.stack(actions),
        torch.stack(states),
    )


def append(store, key, value):
    store.setdefault(key, []).append(value.detach().float().cpu())


def encode_stream(module, values, name):
    """Encode a stream, explicitly adapting legacy checkpoint input mismatch."""
    expected = int(module.patch_embed.in_channels)
    actual = int(values.shape[-1])
    if expected != actual:
        if actual > expected:
            raise ValueError(f"{name} has {actual} channels but checkpoint expects {expected}")
        print(
            f"WARNING: legacy {name} encoder expects {expected} channels but data has "
            f"{actual}; zero-padding for predictor activation probes",
            flush=True,
        )
        values = F.pad(values, (0, expected - actual))
    return module(values)


def collect_activations(
    modules,
    dataset,
    choices,
    batch_size,
    frameskip,
    nframes,
    device,
    include_kinds=None,
):
    """Collect intermediate representations for a set of trajectory windows.

    ``include_kinds`` optionally limits collection to representation suffixes
    such as ``{"cls", "pooled_patches", "projected_aggregate",
    "pooled_visual"}``.  The default keeps the original all-representations
    behavior.  The filter is useful for walkthroughs because retaining every
    individual patch at every layer can require several gigabytes.
    """
    encoder = modules["encoder"]
    predictor = modules["predictor"]
    representations = {}
    all_states, all_actions = [], []
    visual_dim = int(encoder.emb_dim)
    if hasattr(encoder, "agg_mlp"):
        token_count = int(encoder.agg_mlp[0].in_features // visual_dim)
        token_side = int(round(math.sqrt(token_count)))
        encoder_input_size = token_side * int(encoder.patch_size)
    else:
        encoder_input_size = 224

    def requested(kind):
        return include_kinds is None or kind in include_kinds

    with torch.inference_mode():
        for start in range(0, len(choices), batch_size):
            batch_choices = choices[start : start + batch_size]
            visual, proprio, action, state = load_batch(
                dataset, batch_choices, frameskip, nframes
            )
            b, t = visual.shape[:2]
            visual = visual.to(device)
            flat = visual.reshape(b * t, *visual.shape[2:])
            if flat.shape[-2:] != (encoder_input_size, encoder_input_size):
                flat = F.interpolate(
                    flat,
                    size=(encoder_input_size, encoder_input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            layer_outputs = encoder.forward_intermediates(flat)
            for output in layer_outputs:
                layer = output["layer"]
                if requested("cls"):
                    append(representations, f"dino/{layer}/cls", output["cls"].reshape(b, t, -1))
                if requested("pooled_patches"):
                    append(
                        representations,
                        f"dino/{layer}/pooled_patches",
                        output["pooled_patches"].reshape(b, t, -1),
                    )
                if requested("individual_patches"):
                    append(
                        representations,
                        f"dino/{layer}/individual_patches",
                        output["patches"].reshape(b, t, output["patches"].shape[1], -1),
                    )
                if "projected" in output:
                    projected = output["projected"]
                    if projected.ndim == 2:
                        projected = projected.unsqueeze(1)
                    if requested("projected_patches"):
                        append(
                            representations,
                            f"dino/{layer}/projected_patches",
                            projected.reshape(b, t, projected.shape[1], -1),
                        )
                    if requested("projected_aggregate"):
                        append(
                            representations,
                            f"dino/{layer}/projected_aggregate",
                            output["aggregated"].reshape(b, t, -1),
                        )

            # Predictor activations use the exact final encoder representation
            # and normalized action/proprio streams used during training.
            visual_tokens = encoder(flat).reshape(b, t, -1, visual_dim)
            prop_emb = encode_stream(
                modules["proprio_encoder"], proprio.to(device), "proprio"
            )
            act_emb = encode_stream(
                modules["action_encoder"], action.to(device), "action"
            )
            prop_tiled = prop_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
            act_tiled = act_emb.unsqueeze(2).expand(-1, -1, visual_tokens.shape[2], -1)
            z = torch.cat([visual_tokens, prop_tiled, act_tiled], dim=-1)
            hist = min(int(predictor.pos_embedding.shape[1] // z.shape[2]), t - 1)
            pred_input = z[:, :hist].reshape(b, hist * z.shape[2], -1)
            _, pred_layers = predictor(pred_input, return_intermediates=True)
            for layer, activation in enumerate(pred_layers):
                activation = activation.reshape(b, hist, z.shape[2], -1)[..., :visual_dim]
                if requested("pooled_visual"):
                    append(
                        representations,
                        f"predictor/{layer}/pooled_visual",
                        activation.mean(dim=2),
                    )
                if requested("individual_visual_tokens"):
                    append(
                        representations,
                        f"predictor/{layer}/individual_visual_tokens",
                        activation,
                    )

            all_states.append(state.float())
            all_actions.append(action.float())

    return (
        {key: torch.cat(value).numpy() for key, value in representations.items()},
        torch.cat(all_states).numpy(),
        torch.cat(all_actions).numpy(),
    )


def astar_labels(states):
    _, valid = make_occupancy_grid()
    goal = nearest_valid_node((2.8, 0.8), valid)
    counts = astar_step_counts(valid, goal)
    labels = np.empty(states.shape[:2], dtype=np.float64)
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            labels[i, j] = counts[nearest_valid_node(tuple(states[i, j, :2]), valid)]
    return labels


def standardize_fit(x_train, y_train, x_test, ridge):
    x_mean = x_train.mean(0, keepdims=True)
    x_std = x_train.std(0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    xs = (x_train - x_mean) / x_std
    xt = (x_test - x_mean) / x_std
    y = np.asarray(y_train)
    if y.ndim == 1:
        y = y[:, None]
    y_mean = y.mean(0, keepdims=True)
    yc = y - y_mean
    n, d = xs.shape
    if d <= n:
        weight = np.linalg.solve(xs.T @ xs + ridge * np.eye(d), xs.T @ yc)
    else:
        weight = xs.T @ np.linalg.solve(xs @ xs.T + ridge * np.eye(n), yc)
    pred = xt @ weight + y_mean
    return pred.squeeze(-1) if pred.shape[1] == 1 else pred, weight


def r2_score(y, pred):
    y = np.asarray(y)
    denominator = np.square(y - y.mean(axis=0)).sum()
    return float(1.0 - np.square(y - pred).sum() / max(denominator, 1e-12))


def binary_auc(y, score):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def subspace_overlap(a, b):
    qa, _ = np.linalg.qr(np.atleast_2d(a))
    qb, _ = np.linalg.qr(np.atleast_2d(b))
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.square(singular).mean())


def flatten_representation(rep, window_indices, transition=False, max_patch_rows=50000):
    selected = rep[window_indices]
    if transition:
        selected = selected[:, 1:] - selected[:, :-1]
    if selected.ndim == 4:
        selected = selected.reshape(-1, selected.shape[-1])
        if len(selected) > max_patch_rows:
            stride = int(math.ceil(len(selected) / max_patch_rows))
            selected = selected[::stride]
        return selected, "patch"
    return selected.reshape(-1, selected.shape[-1]), "frame"


def repeat_labels(labels, rep, window_indices, transition=False, target_rows=None):
    selected = labels[window_indices]
    if transition:
        selected = selected[:, : rep.shape[1] - 1]
    else:
        selected = selected[:, : rep.shape[1]]
    flat = selected.reshape(-1, *selected.shape[2:])
    if rep.ndim == 4:
        flat = np.repeat(flat, rep.shape[2], axis=0)
    if target_rows is not None and len(flat) > target_rows:
        stride = int(math.ceil(len(flat) / target_rows))
        flat = flat[::stride][:target_rows]
    return flat


def probe_representation(name, rep, states, actions, astar, train_idx, test_idx, ridge):
    usable_t = rep.shape[1]
    xy_delta = states[:, 1:usable_t, :2] - states[:, : usable_t - 1, :2]
    speed = np.linalg.norm(xy_delta, axis=-1)
    direction = xy_delta / np.maximum(speed[..., None], 1e-6)
    action = actions[:, : usable_t - 1, :]
    action_mag = np.linalg.norm(action, axis=-1)
    moving_action = action_mag > np.quantile(action_mag, 0.5)
    collision = moving_action & (speed < np.quantile(speed, 0.15))

    xs_train, _ = flatten_representation(rep, train_idx)
    xs_test, _ = flatten_representation(rep, test_idx)
    astar_train = repeat_labels(astar, rep, train_idx, target_rows=len(xs_train))
    astar_test = repeat_labels(astar, rep, test_idx, target_rows=len(xs_test))
    astar_pred, astar_weight = standardize_fit(xs_train, astar_train, xs_test, ridge)

    xt_train, _ = flatten_representation(rep, train_idx, transition=True)
    xt_test, _ = flatten_representation(rep, test_idx, transition=True)
    speed_train = repeat_labels(speed, rep, train_idx, transition=True, target_rows=len(xt_train))
    speed_test = repeat_labels(speed, rep, test_idx, transition=True, target_rows=len(xt_test))
    direction_train = repeat_labels(direction, rep, train_idx, transition=True, target_rows=len(xt_train))
    direction_test = repeat_labels(direction, rep, test_idx, transition=True, target_rows=len(xt_test))
    mag_train = repeat_labels(action_mag, rep, train_idx, transition=True, target_rows=len(xt_train))
    mag_test = repeat_labels(action_mag, rep, test_idx, transition=True, target_rows=len(xt_test))
    action_train = repeat_labels(action, rep, train_idx, transition=True, target_rows=len(xt_train))
    action_test = repeat_labels(action, rep, test_idx, transition=True, target_rows=len(xt_test))
    collision_train = repeat_labels(collision, rep, train_idx, transition=True, target_rows=len(xt_train))
    collision_test = repeat_labels(collision, rep, test_idx, transition=True, target_rows=len(xt_test))

    speed_pred, speed_weight = standardize_fit(xt_train, speed_train, xt_test, ridge)
    direction_pred, direction_weight = standardize_fit(xt_train, direction_train, xt_test, ridge)
    mag_pred, mag_weight = standardize_fit(xt_train, mag_train, xt_test, ridge)
    action_pred, action_decode_weight = standardize_fit(xt_train, action_train, xt_test, ridge)
    collision_score, collision_weight = standardize_fit(
        xt_train, collision_train.astype(float), xt_test, ridge
    )
    direction_cos = np.sum(direction_pred * direction_test, axis=-1) / (
        np.linalg.norm(direction_pred, axis=-1) * np.linalg.norm(direction_test, axis=-1) + 1e-6
    )

    # Forward dynamics: action -> latent change. Its coefficient columns span
    # the locally action-controllable directions in this representation.
    controllable_pred, controllability_weight = standardize_fit(
        action_train, xt_train, action_test, ridge
    )
    astar_vector = astar_weight[:, :1]
    controllable_basis = controllability_weight.T
    overlap = subspace_overlap(astar_vector, controllable_basis)

    metrics = {
        "representation": name,
        "n_train": int(len(xt_train)),
        "feature_dim": int(xt_train.shape[1]),
        "astar_spearman": float(spearmanr(astar_test, astar_pred).statistic),
        "astar_r2": r2_score(astar_test, astar_pred),
        "speed_r2": r2_score(speed_test, speed_pred),
        "direction_cosine": float(np.nanmean(direction_cos)),
        "collision_accuracy": float(np.mean((collision_score >= 0.5) == collision_test)),
        "collision_auc": binary_auc(collision_test, collision_score),
        "action_magnitude_r2": r2_score(mag_test, mag_pred),
        "action_decode_r2": r2_score(action_test, action_pred),
        "action_to_latent_r2": r2_score(xt_test, controllable_pred),
        "astar_action_subspace_overlap": overlap,
        "speed_action_subspace_overlap": subspace_overlap(speed_weight[:, :1], controllable_basis),
        "direction_action_subspace_overlap": subspace_overlap(direction_weight, controllable_basis),
    }
    metrics["selection_score"] = float(
        np.nanmean(
            [
                max(metrics["astar_spearman"], 0.0),
                max(metrics["speed_r2"], 0.0),
                max(metrics["direction_cosine"], 0.0),
                max(metrics["action_decode_r2"], 0.0),
                max(2 * metrics["collision_auc"] - 1, 0.0),
            ]
        )
    )
    return metrics


def select_layer(rows, family):
    candidates = [row for row in rows if row["representation"].startswith(family + "/")]
    best = max(candidates, key=lambda row: row["selection_score"])
    _, layer, representation = best["representation"].split("/", 2)
    return {"layer": int(layer), "representation": representation, "score": best["selection_score"]}


def write_artifacts(output_dir, rows, selection, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    (output_dir / "selected_layers.json").write_text(json.dumps(selection, indent=2))
    with (output_dir / "layer_probe_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    top = sorted(rows, key=lambda row: row["selection_score"], reverse=True)[:12]
    lines = [
        "# UMaze layerwise physics probes",
        "",
        "These probes ask what information is *linearly readable* from each representation. "
        "They do not by themselves prove causal use by the planner.",
        "",
        f"- Windows: {args.max_windows}; frame skip: {args.frameskip}; split is by window.",
        "- Collision/stall means above-median commanded action with bottom-15% physical displacement.",
        "- `action_to_latent_r2` fits a local action→latent-change map; its column space is the action-controllable subspace.",
        "- `astar_action_subspace_overlap` is the squared principal-cosine overlap between the A* probe and that controllable subspace.",
        "- Layer selection averages non-negative A* rank correlation, speed R², direction cosine, action decode R², and collision AUC above chance.",
        "",
        "## Selected layers",
        "",
        f"- DINO: layer {selection['dino']['layer']} ({selection['dino']['representation']}, score {selection['dino']['score']:.3f})",
        f"- Predictor: layer {selection['predictor']['layer']} ({selection['predictor']['representation']}, score {selection['predictor']['score']:.3f})",
        "",
        "## Highest-scoring representations",
        "",
        "| representation | score | A* ρ | speed R² | direction cos | collision AUC | action R² | A*↔control overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row['representation']} | {row['selection_score']:.3f} | "
            f"{row['astar_spearman']:.3f} | {row['speed_r2']:.3f} | "
            f"{row['direction_cosine']:.3f} | {row['collision_auc']:.3f} | "
            f"{row['action_decode_r2']:.3f} | {row['astar_action_subspace_overlap']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A high A* score means latent ordering agrees with true maze progress. A high speed or direction score means physical motion is easy to recover. High action→latent R² means commands move the representation predictably. Overlap tells us whether maze-distance information lies in directions the action-conditioned predictor can actually manipulate; overlap that is too low suggests planning-relevant geometry is present but dynamically inaccessible.",
        "",
        "The selected layers are the inputs to the layer-aware factorized ablation. The full table remains the primary result; selection should be treated as a preregistered heuristic rather than post-hoc proof.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for ax, family in zip(axes, ("dino", "predictor")):
            family_rows = [row for row in rows if row["representation"].startswith(family + "/")]
            for representation in sorted({row["representation"].split("/", 2)[2] for row in family_rows}):
                subset = [row for row in family_rows if row["representation"].endswith("/" + representation)]
                subset.sort(key=lambda row: int(row["representation"].split("/")[1]))
                ax.plot(
                    [int(row["representation"].split("/")[1]) for row in subset],
                    [row["selection_score"] for row in subset],
                    marker="o",
                    label=representation,
                )
            ax.set(title=family.upper(), xlabel="layer", ylabel="selection score")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.savefig(output_dir / "layer_selection_scores.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    modules = load_checkpoint(args.checkpoint, device)
    use_frame_files = (args.data_dir / "obses" / "episode_000_frame_000.pth").exists()
    dataset = PointMazeDataset(
        data_path=str(args.data_dir),
        transform=default_transform(224),
        normalize_action=True,
        use_frame_files=use_frame_files,
    )
    choices = sample_windows(
        dataset, args.max_windows, args.frameskip, args.num_frames, args.seed
    )
    representations, states, actions = collect_activations(
        modules,
        dataset,
        choices,
        args.batch_size,
        args.frameskip,
        args.num_frames,
        device,
    )
    astar = astar_labels(states)
    permutation = np.random.default_rng(args.seed).permutation(len(states))
    split = max(1, int(0.8 * len(permutation)))
    train_idx, test_idx = permutation[:split], permutation[split:]
    rows = []
    for name, representation in sorted(representations.items()):
        rows.append(
            probe_representation(
                name, representation, states, actions, astar, train_idx, test_idx, args.ridge
            )
        )
        print(name, rows[-1]["selection_score"], flush=True)
    selection = {
        "dino": select_layer(rows, "dino"),
        "predictor": select_layer(rows, "predictor"),
    }
    write_artifacts(args.output_dir, rows, selection, args)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
