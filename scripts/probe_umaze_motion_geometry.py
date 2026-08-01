#!/usr/bin/env python3
"""Probe the geometry of physical motion in trained UMaze representations.

This complements ``probe_umaze_layers.py``.  It uses identical held-out
trajectory windows, but asks whether position, velocity, and acceleration are
linearly readable in Cartesian and polar coordinates.  It also measures
motion-subspace overlap, the effective dimensionality of heading, and spatial
redundancy of heading information across DINO patches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from datasets.img_transforms import default_transform
from datasets.point_maze_dset import PointMazeDataset
from scripts.probe_umaze_layers import (
    collect_activations,
    load_checkpoint,
    r2_score,
    sample_windows,
    standardize_fit,
    subspace_overlap,
)


GOAL_XY = np.asarray([2.8, 0.8], dtype=np.float64)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selected-layers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--direction-harmonics", type=int, default=16)
    parser.add_argument("--patch-samples", type=int, default=25)
    return parser.parse_args()


def unit_vectors(vectors: np.ndarray, eps: float = 1e-6):
    magnitude = np.linalg.norm(vectors, axis=-1)
    return vectors / np.maximum(magnitude[..., None], eps), magnitude


def circular_harmonics(unit_direction: np.ndarray, harmonics: int):
    theta = np.arctan2(unit_direction[..., 1], unit_direction[..., 0])
    return np.stack(
        [fn(k * theta) for k in range(1, harmonics + 1) for fn in (np.cos, np.sin)],
        axis=-1,
    )


def effective_rank(values: np.ndarray) -> tuple[float, int, list[float]]:
    singular = np.linalg.svd(values, compute_uv=False)
    energy = np.square(singular)
    if energy.sum() <= 1e-12:
        return 0.0, 0, singular.tolist()
    probability = energy / energy.sum()
    rank = float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))
    d90 = int(np.searchsorted(np.cumsum(probability), 0.9) + 1)
    return rank, d90, singular.tolist()


def principal_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(np.atleast_2d(a))
    qb, _ = np.linalg.qr(np.atleast_2d(b))
    largest_cosine = float(np.linalg.svd(qa.T @ qb, compute_uv=False).max())
    return float(np.degrees(np.arccos(np.clip(largest_cosine, -1.0, 1.0))))


def physical_targets(states: np.ndarray, frameskip: int):
    position = states[..., :2].astype(np.float64)
    if states.shape[-1] >= 4:
        velocity = states[..., 2:4].astype(np.float64)
    else:
        velocity = np.zeros_like(position)
        velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) / max(frameskip, 1)
        velocity[:, 0] = velocity[:, 1]
    acceleration = np.full_like(velocity, np.nan)
    acceleration[:, 1:] = (velocity[:, 1:] - velocity[:, :-1]) / max(frameskip, 1)

    heading, speed = unit_vectors(velocity)
    acceleration_direction, acceleration_magnitude = unit_vectors(np.nan_to_num(acceleration))
    acceleration_direction[:, 0] = np.nan
    acceleration_magnitude[:, 0] = np.nan
    goal_delta = GOAL_XY - position
    goal_bearing, goal_range = unit_vectors(goal_delta)
    return {
        "position_xy": position,
        "goal_delta_xy": goal_delta,
        "goal_range": goal_range,
        "goal_bearing": goal_bearing,
        "velocity_xy": velocity,
        "speed": speed,
        "heading": heading,
        "acceleration_xy": acceleration,
        "acceleration_magnitude": acceleration_magnitude,
        "acceleration_direction": acceleration_direction,
    }


def flatten_frames(rep: np.ndarray, indices: np.ndarray, times: slice | None = None):
    selected = rep[indices]
    if times is not None:
        selected = selected[:, times]
    if selected.ndim == 4:
        selected = selected.mean(axis=2)
    return selected.reshape(-1, selected.shape[-1])


def flatten_target(target: np.ndarray, indices: np.ndarray, times: slice | None = None):
    selected = target[indices]
    if times is not None:
        selected = selected[:, times]
    return selected.reshape(-1, *selected.shape[2:])


def direction_cosine(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray):
    target = target[mask]
    prediction = prediction[mask]
    cosine = np.sum(target * prediction, axis=-1) / (
        np.linalg.norm(target, axis=-1) * np.linalg.norm(prediction, axis=-1) + 1e-8
    )
    return float(np.nanmean(cosine))


def angular_mae_degrees(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray):
    target_theta = np.arctan2(target[mask, 1], target[mask, 0])
    pred_theta = np.arctan2(prediction[mask, 1], prediction[mask, 0])
    delta = np.arctan2(np.sin(pred_theta - target_theta), np.cos(pred_theta - target_theta))
    return float(np.degrees(np.mean(np.abs(delta))))


def fit_target(x_train, y_train, x_test, y_test, ridge):
    finite_train = np.isfinite(y_train).all(axis=-1) if y_train.ndim > 1 else np.isfinite(y_train)
    finite_test = np.isfinite(y_test).all(axis=-1) if y_test.ndim > 1 else np.isfinite(y_test)
    prediction, weight = standardize_fit(
        x_train[finite_train], y_train[finite_train], x_test[finite_test], ridge
    )
    return y_test[finite_test], prediction, weight, finite_test


def probe_representation(name, rep, targets, actions, train_idx, test_idx, ridge, harmonics):
    usable_t = rep.shape[1]
    x_train = flatten_frames(rep, train_idx)
    x_test = flatten_frames(rep, test_idx)
    labels_train = {k: flatten_target(v[:, :usable_t], train_idx) for k, v in targets.items()}
    labels_test = {k: flatten_target(v[:, :usable_t], test_idx) for k, v in targets.items()}

    weights = {}
    scores = {"representation": name, "feature_dim": int(x_train.shape[1])}
    for label in ("position_xy", "goal_delta_xy", "velocity_xy", "acceleration_xy"):
        truth, prediction, weights[label], _ = fit_target(
            x_train, labels_train[label], x_test, labels_test[label], ridge
        )
        scores[f"{label}_r2"] = r2_score(truth, prediction)
        scores[f"{label}_rmse"] = float(np.sqrt(np.mean(np.square(truth - prediction))))

    for label in ("goal_range", "speed", "acceleration_magnitude"):
        truth, prediction, weights[label], _ = fit_target(
            x_train, labels_train[label], x_test, labels_test[label], ridge
        )
        scores[f"{label}_r2"] = r2_score(truth, prediction)
        scores[f"{label}_mae"] = float(np.mean(np.abs(truth - prediction)))

    for label, magnitude_label in (
        ("goal_bearing", "goal_range"),
        ("heading", "speed"),
        ("acceleration_direction", "acceleration_magnitude"),
    ):
        truth, prediction, weights[label], finite = fit_target(
            x_train, labels_train[label], x_test, labels_test[label], ridge
        )
        magnitude = labels_test[magnitude_label][finite]
        threshold = max(float(np.quantile(magnitude, 0.1)), 1e-6)
        moving = magnitude > threshold
        scores[f"{label}_cosine"] = direction_cosine(truth, prediction, moving)
        scores[f"{label}_angular_mae_deg"] = angular_mae_degrees(truth, prediction, moving)

    action_train = flatten_target(actions[:, :usable_t], train_idx)
    action_test = flatten_target(actions[:, :usable_t], test_idx)
    _, _, weights["action"], _ = fit_target(
        x_train, action_train, x_test, action_test, ridge
    )

    moving_train = labels_train["speed"] > max(
        float(np.quantile(labels_train["speed"], 0.1)), 1e-6
    )
    direction_bank = circular_harmonics(labels_train["heading"][moving_train], harmonics)
    _, heading_bank_weight = standardize_fit(
        x_train[moving_train], direction_bank, x_train[moving_train], ridge
    )
    rank, d90, spectrum = effective_rank(heading_bank_weight)
    scores["heading_harmonic_effective_rank"] = rank
    scores["heading_harmonic_dimensions_90pct"] = d90
    scores["heading_harmonic_singular_values"] = spectrum

    pairs = (
        ("speed", "heading"),
        ("heading", "acceleration_xy"),
        ("heading", "position_xy"),
        ("heading", "goal_delta_xy"),
        ("velocity_xy", "acceleration_xy"),
        ("position_xy", "velocity_xy"),
        ("goal_delta_xy", "action"),
        ("speed", "action"),
        ("heading", "action"),
    )
    for left, right in pairs:
        key = f"{left}_vs_{right}"
        scores[f"{key}_overlap"] = subspace_overlap(weights[left], weights[right])
        scores[f"{key}_min_angle_deg"] = principal_angle_degrees(weights[left], weights[right])
    return scores


def choose_representations(representations, selected, reference):
    dino_layer = str(selected["dino"]["layer"])
    predictor_layer = str(selected["predictor"]["layer"])
    chosen = {}
    for name, rep in representations.items():
        family, layer, kind = name.split("/", 2)
        at_selected = (family == "dino" and layer == dino_layer) or (
            family == "predictor" and layer == predictor_layer
        )
        layerwise_reference = reference and kind in ("pooled_patches", "pooled_visual")
        if at_selected or layerwise_reference:
            chosen[name] = rep
    return chosen


def sampled_patch_indices(patch_count: int, count: int):
    if count >= patch_count:
        return np.arange(patch_count)
    side = int(round(math.sqrt(patch_count)))
    sample_side = max(2, int(round(math.sqrt(count))))
    coordinates = np.linspace(0, side - 1, sample_side).round().astype(int)
    return np.unique([row * side + col for row in coordinates for col in coordinates])


def patch_redundancy(rep, targets, train_idx, test_idx, ridge, max_patches):
    indices = sampled_patch_indices(rep.shape[2], max_patches)
    train_heading = flatten_target(targets["heading"][:, : rep.shape[1]], train_idx)
    test_heading = flatten_target(targets["heading"][:, : rep.shape[1]], test_idx)
    train_speed = flatten_target(targets["speed"][:, : rep.shape[1]], train_idx)
    test_speed = flatten_target(targets["speed"][:, : rep.shape[1]], test_idx)
    train_mask = train_speed > max(float(np.quantile(train_speed, 0.1)), 1e-6)
    test_mask = test_speed > max(float(np.quantile(test_speed, 0.1)), 1e-6)
    predictions, cosines = [], []
    for patch in indices:
        x_train = rep[train_idx, :, patch].reshape(-1, rep.shape[-1])
        x_test = rep[test_idx, :, patch].reshape(-1, rep.shape[-1])
        prediction, _ = standardize_fit(
            x_train[train_mask], train_heading[train_mask], x_test[test_mask], ridge
        )
        predictions.append(prediction)
        cosines.append(direction_cosine(test_heading[test_mask], prediction, np.ones(len(prediction), bool)))
    prediction_vectors = [value.reshape(-1) for value in predictions]
    pairwise = []
    for i in range(len(prediction_vectors)):
        for j in range(i + 1, len(prediction_vectors)):
            a, b = prediction_vectors[i], prediction_vectors[j]
            pairwise.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
    return {
        "sampled_patch_indices": indices.tolist(),
        "per_patch_heading_cosine": [float(value) for value in cosines],
        "mean_patch_heading_cosine": float(np.mean(cosines)),
        "best_patch_heading_cosine": float(np.max(cosines)),
        "mean_pairwise_prediction_cosine": float(np.mean(pairwise)),
        "fraction_within_95pct_of_best": float(np.mean(np.asarray(cosines) >= 0.95 * np.max(cosines))),
    }


def scalar_rows(rows):
    return [
        {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for row in rows
    ]


def write_artifacts(output_dir, rows, patch_rows, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"probe_metrics": rows, "patch_redundancy": patch_rows, "config": vars(args)}
    payload["config"]["data_dir"] = str(args.data_dir)
    payload["config"]["selected_layers"] = str(args.selected_layers)
    payload["config"]["output_dir"] = str(args.output_dir)
    payload["config"]["checkpoint"] = [[label, str(path)] for label, path in args.checkpoint]
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    table = scalar_rows(rows)
    with (output_dir / "motion_probe_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    selected_rows = [row for row in rows if row.get("selected_layer")]
    lines = [
        "# UMaze Cartesian/polar motion-geometry probes",
        "",
        "This post-training study tests whether physical variables are linearly readable, not merely correlated with planning success. All probes use the same held-out trajectory-window split and the same ridge penalty.",
        "",
        "## Questions",
        "",
        "1. Is position readable as `(x,y)`, and is goal-relative position cleaner as range/bearing?",
        "2. Is velocity cleaner as Cartesian `(vx,vy)` or polar `(speed, heading)`?",
        "3. Does acceleration become readable at the same layer as velocity?",
        "4. Are direction and speed/other intuitive-physics probe subspaces close to orthogonal?",
        "5. How many linearly independent heading harmonics are supported, and is heading spatially redundant across patches?",
        "",
        "Direction is represented as `(cos θ, sin θ)`, avoiding the discontinuity at ±π. Acceleration is the finite difference of the dataset's physical velocity. Direction metrics exclude the slowest 10% of samples, where angle is ill-defined.",
        "",
        "## Selected-layer checkpoint comparison",
        "",
        "| condition | representation | pos R² | vel R² | speed R² | heading cos | accel R² | accel cos | speed↔heading overlap | min angle | heading d90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['condition']} | {row['representation']} | {row['position_xy_r2']:.3f} | "
            f"{row['velocity_xy_r2']:.3f} | {row['speed_r2']:.3f} | {row['heading_cosine']:.3f} | "
            f"{row['acceleration_xy_r2']:.3f} | {row['acceleration_direction_cosine']:.3f} | "
            f"{row['speed_vs_heading_overlap']:.3f} | {row['speed_vs_heading_min_angle_deg']:.1f}° | "
            f"{row['heading_harmonic_dimensions_90pct']} |"
        )
    lines += [
        "",
        "## Reading the results",
        "",
        "- Higher R²/cosine and lower RMSE/angular error mean easier linear decoding.",
        "- Polar dominance means strong speed R² and heading cosine even when joint Cartesian velocity R² is weaker.",
        "- Similar velocity and acceleration layer-onset supports the paper's claim that acceleration can emerge without a separate intermediate velocity stage.",
        "- Low subspace overlap and a large principal angle mean the two variables occupy distinct directions in feature space.",
        "- `heading_harmonic_dimensions_90pct` is the number of singular directions needed for 90% of a 32-target circular-harmonic probe's weight energy; it is a diagnostic of distributed direction coding, not the intrinsic dimension of the complete representation.",
        "- High pairwise patch prediction cosine and many patches near the best patch indicate spatially redundant heading information.",
        "",
        "These are diagnostic associations. A variable being decodable does not prove that the predictor or planner causally uses it.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt

        layerwise = [row for row in rows if row["condition"] == args.checkpoint[0][0]]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for ax, metric, title in (
            (axes[0], "velocity_xy_r2", "Cartesian velocity"),
            (axes[1], "speed_r2", "Polar speed"),
            (axes[2], "acceleration_xy_r2", "Cartesian acceleration"),
        ):
            for family in ("dino", "predictor"):
                subset = [r for r in layerwise if r["representation"].startswith(family + "/") and ("pooled" in r["representation"])]
                subset.sort(key=lambda r: int(r["representation"].split("/")[1]))
                if subset:
                    ax.plot([int(r["representation"].split("/")[1]) for r in subset], [r[metric] for r in subset], marker="o", label=family)
            ax.set(title=title, xlabel="layer", ylabel=metric)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.savefig(output_dir / "motion_emergence_by_layer.png", dpi=180)
        plt.close(fig)

        comparison = selected_rows
        labels = [f"{r['condition']}\n{r['representation'].split('/')[-1]}" for r in comparison]
        x = np.arange(len(comparison))
        fig, ax = plt.subplots(figsize=(max(10, len(comparison) * 0.7), 5), constrained_layout=True)
        width = 0.18
        for offset, metric, label in (
            (-1.5, "velocity_xy_r2", "velocity xy R²"),
            (-0.5, "speed_r2", "speed R²"),
            (0.5, "heading_cosine", "heading cosine"),
            (1.5, "acceleration_xy_r2", "acceleration xy R²"),
        ):
            ax.bar(x + offset * width, [r[metric] for r in comparison], width, label=label)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("held-out score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2)
        fig.savefig(output_dir / "cartesian_vs_polar_by_condition.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (output_dir / "plot_error.txt").write_text(str(exc))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    selected = json.loads(args.selected_layers.read_text())
    use_frame_files = (args.data_dir / "obses" / "episode_000_frame_000.pth").exists()
    dataset = PointMazeDataset(
        data_path=str(args.data_dir),
        transform=default_transform(224),
        normalize_action=True,
        use_frame_files=use_frame_files,
    )
    choices = sample_windows(dataset, args.max_windows, args.frameskip, args.num_frames, args.seed)
    permutation = np.random.default_rng(args.seed).permutation(len(choices))
    split = max(1, int(0.8 * len(permutation)))
    train_idx, test_idx = permutation[:split], permutation[split:]
    rows, patch_rows = [], []

    for checkpoint_index, (condition, checkpoint) in enumerate(args.checkpoint):
        print(f"loading {condition}: {checkpoint}", flush=True)
        modules = load_checkpoint(checkpoint, device)
        representations, states, actions = collect_activations(
            modules, dataset, choices, args.batch_size, args.frameskip, args.num_frames, device
        )
        targets = physical_targets(states, args.frameskip)
        chosen = choose_representations(representations, selected, checkpoint_index == 0)
        for name, representation in sorted(chosen.items()):
            row = probe_representation(
                name, representation, targets, actions, train_idx, test_idx, args.ridge,
                args.direction_harmonics,
            )
            row["condition"] = condition
            family, layer, _ = name.split("/", 2)
            row["selected_layer"] = bool(
                (family == "dino" and int(layer) == selected["dino"]["layer"])
                or (family == "predictor" and int(layer) == selected["predictor"]["layer"])
            )
            rows.append(row)
            print(condition, name, row["velocity_xy_r2"], row["speed_r2"], flush=True)

            if row["selected_layer"] and name.endswith("/individual_patches"):
                patch_rows.append({
                    "condition": condition,
                    "representation": name,
                    **patch_redundancy(
                        representation, targets, train_idx, test_idx, args.ridge,
                        args.patch_samples,
                    ),
                })
        del modules, representations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_artifacts(args.output_dir, rows, patch_rows, args)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
