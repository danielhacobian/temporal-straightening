#!/usr/bin/env python3
"""Reusable analysis helpers for the UMaze layerwise probe walkthrough.

The expensive model-specific activation collection remains in
``probe_umaze_layers.py``.  This module contains NumPy-only alignment, split,
linear-probe, control, cache, and uncertainty utilities so the notebook can be
tested without a checkpoint or GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def r2_score(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = np.square(y - y.mean(axis=0)).sum()
    return float(1.0 - np.square(y - prediction).sum() / max(denominator, 1e-12))


def standardize_fit(x_train, y_train, x_test, ridge):
    """NumPy ridge implementation matching ``probe_umaze_layers.py``."""
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
    prediction = xt @ weight + y_mean
    return (
        prediction.squeeze(-1) if prediction.shape[1] == 1 else prediction,
        weight,
    )


def build_motion_targets(states: np.ndarray, frameskip: int, step_dt: float = 1.0):
    """Create framewise and transitionwise Cartesian/polar motion targets.

    State channels 2:4 are used as instantaneous velocity when available.
    Transition velocity always comes from displacement, making it the matching
    label for a temporal feature difference spanning two sampled frames.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[-1] < 2:
        raise ValueError("states must have shape [window, time, >=2]")
    dt = float(frameskip) * float(step_dt)
    if dt <= 0:
        raise ValueError("frameskip * step_dt must be positive")

    position = states[..., :2]
    transition_velocity = np.diff(position, axis=1) / dt
    if states.shape[-1] >= 4:
        velocity = states[..., 2:4]
    else:
        velocity = np.empty_like(position)
        velocity[:, :-1] = transition_velocity
        velocity[:, -1] = transition_velocity[:, -1]

    acceleration = np.full_like(velocity, np.nan)
    acceleration[:, 1:] = np.diff(velocity, axis=1) / dt
    transition_acceleration = np.diff(transition_velocity, axis=1) / dt

    def polar(vector):
        magnitude = np.linalg.norm(vector, axis=-1)
        direction = vector / np.maximum(magnitude[..., None], 1e-8)
        return magnitude, direction

    speed, heading = polar(velocity)
    transition_speed, transition_heading = polar(transition_velocity)
    acceleration_magnitude, acceleration_direction = polar(np.nan_to_num(acceleration))
    acceleration_direction[:, 0] = np.nan
    transition_acceleration_magnitude, transition_acceleration_direction = polar(
        transition_acceleration
    )
    return {
        "position": position,
        "velocity": velocity,
        "speed": speed,
        "heading": heading,
        "acceleration": acceleration,
        "acceleration_magnitude": acceleration_magnitude,
        "acceleration_direction": acceleration_direction,
        "transition_velocity": transition_velocity,
        "transition_speed": transition_speed,
        "transition_heading": transition_heading,
        "transition_acceleration": transition_acceleration,
        "transition_acceleration_magnitude": transition_acceleration_magnitude,
        "transition_acceleration_direction": transition_acceleration_direction,
        "dt": dt,
    }


def align_representation(rep: np.ndarray, targets: dict, variable: str, mode: str):
    """Align a representation tensor and physical target without flattening windows.

    Returns ``(features, labels, position_context)``.  Position context is the
    location matched to each label and supports position-only and residualized
    controls.
    """
    rep = np.asarray(rep)
    if rep.ndim == 4:
        rep = rep.mean(axis=2)
    if rep.ndim != 3:
        raise ValueError("representation must have shape [window, time, feature]")
    t = rep.shape[1]
    position = targets["position"][:, :t]

    frame_targets = {
        "position": targets["position"][:, :t],
        "velocity": targets["velocity"][:, :t],
        "speed": targets["speed"][:, :t],
        "heading": targets["heading"][:, :t],
        "acceleration": targets["acceleration"][:, :t],
        "acceleration_magnitude": targets["acceleration_magnitude"][:, :t],
        "acceleration_direction": targets["acceleration_direction"][:, :t],
    }
    transition_targets = {
        "velocity": targets["transition_velocity"][:, : max(t - 1, 0)],
        "speed": targets["transition_speed"][:, : max(t - 1, 0)],
        "heading": targets["transition_heading"][:, : max(t - 1, 0)],
    }
    second_targets = {
        "acceleration": targets["transition_acceleration"][:, : max(t - 2, 0)],
        "acceleration_magnitude": targets["transition_acceleration_magnitude"][:, : max(t - 2, 0)],
        "acceleration_direction": targets["transition_acceleration_direction"][:, : max(t - 2, 0)],
    }

    if mode == "frame":
        if variable not in frame_targets:
            raise ValueError(f"{variable!r} has no framewise target")
        return rep, frame_targets[variable], position
    if mode in ("delta", "concat"):
        if variable not in transition_targets:
            raise ValueError(f"{mode} is only defined for velocity/speed/heading")
        if t < 2:
            raise ValueError("at least two representation slots are required")
        features = np.diff(rep, axis=1) if mode == "delta" else np.concatenate(
            [rep[:, :-1], rep[:, 1:]], axis=-1
        )
        context = 0.5 * (position[:, :-1] + position[:, 1:])
        return features, transition_targets[variable], context
    if mode in ("second_delta", "concat3"):
        if variable not in second_targets:
            raise ValueError(f"{mode} is only defined for acceleration targets")
        if t < 3:
            raise ValueError("at least three representation slots are required")
        features = (
            rep[:, 2:] - 2.0 * rep[:, 1:-1] + rep[:, :-2]
            if mode == "second_delta"
            else np.concatenate([rep[:, :-2], rep[:, 1:-1], rep[:, 2:]], axis=-1)
        )
        return features, second_targets[variable], position[:, 1:-1]
    raise ValueError(f"unknown mode {mode!r}")


def episode_group_split(choices, test_fraction: float = 0.2, seed: int = 0):
    """Split windows by episode so one trajectory never appears on both sides."""
    choices = np.asarray(choices, dtype=np.int64)
    if choices.ndim != 2 or choices.shape[1] < 1:
        raise ValueError("choices must contain [episode, start] rows")
    episodes = np.unique(choices[:, 0])
    if len(episodes) < 2:
        raise ValueError("episode-held-out evaluation requires at least two episodes")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_test = min(len(episodes) - 1, max(1, int(math.ceil(test_fraction * len(episodes)))))
    test_episodes = set(shuffled[:n_test].tolist())
    test = np.asarray([i for i, episode in enumerate(choices[:, 0]) if episode in test_episodes])
    train = np.asarray([i for i, episode in enumerate(choices[:, 0]) if episode not in test_episodes])
    return train, test


def spatial_holdout_split(
    anchor_position: np.ndarray,
    axis: int = 1,
    quantile: float = 0.8,
    high: bool = True,
    buffer_fraction: float = 0.05,
):
    """Hold out one spatial tail and drop a buffer band around its boundary."""
    anchor_position = np.asarray(anchor_position)
    coordinate = anchor_position[:, axis]
    boundary = float(np.quantile(coordinate, quantile if high else 1.0 - quantile))
    buffer = float(np.ptp(coordinate) * buffer_fraction)
    if high:
        train = np.flatnonzero(coordinate < boundary - buffer)
        test = np.flatnonzero(coordinate >= boundary)
    else:
        train = np.flatnonzero(coordinate > boundary + buffer)
        test = np.flatnonzero(coordinate <= boundary)
    if not len(train) or not len(test):
        raise ValueError("spatial split produced an empty train or test set")
    return train, test, {"boundary": boundary, "buffer": buffer, "axis": axis, "high": high}


def _flatten_valid(features, labels, window_indices):
    x = np.asarray(features)[window_indices].reshape(-1, features.shape[-1])
    y = np.asarray(labels)[window_indices]
    y = y.reshape(-1, *y.shape[2:])
    finite = np.isfinite(x).all(axis=-1)
    finite &= np.isfinite(y).all(axis=-1) if y.ndim > 1 else np.isfinite(y)
    return x[finite], y[finite]


def fit_probe(features, labels, train_idx, test_idx, ridge: float = 10.0):
    """Fit the standardized ridge probe used by the existing UMaze analysis."""
    x_train, y_train = _flatten_valid(features, labels, train_idx)
    x_test, y_test = _flatten_valid(features, labels, test_idx)
    prediction, weight = standardize_fit(x_train, y_train, x_test, ridge)
    if y_test.ndim == 2 and y_test.shape[1] == 1 and prediction.ndim == 1:
        y_test = y_test[:, 0]
    return y_test, prediction, weight


def regression_scores(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return {
        "r2": r2_score(truth, prediction),
        "rmse": float(np.sqrt(np.mean(np.square(truth - prediction)))),
        "mae": float(np.mean(np.abs(truth - prediction))),
    }


def direction_scores(truth, prediction, minimum_magnitude=None):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    mask = np.isfinite(truth).all(axis=-1) & np.isfinite(prediction).all(axis=-1)
    if minimum_magnitude is not None:
        mask &= np.linalg.norm(truth, axis=-1) >= minimum_magnitude
    truth, prediction = truth[mask], prediction[mask]
    cosine = np.sum(truth * prediction, axis=-1) / (
        np.linalg.norm(truth, axis=-1) * np.linalg.norm(prediction, axis=-1) + 1e-8
    )
    true_theta = np.arctan2(truth[:, 1], truth[:, 0])
    pred_theta = np.arctan2(prediction[:, 1], prediction[:, 0])
    delta = np.arctan2(np.sin(pred_theta - true_theta), np.cos(pred_theta - true_theta))
    return {
        "cosine": float(np.mean(cosine)),
        "angular_mae_deg": float(np.degrees(np.mean(np.abs(delta)))),
    }


def mask_slow_directions(labels, magnitude, train_idx, quantile: float = 0.1):
    """Mark low-magnitude direction labels NaN using a training-only cutoff."""
    labels = np.asarray(labels, dtype=float).copy()
    magnitude = np.asarray(magnitude)
    cutoff = float(np.nanquantile(magnitude[train_idx], quantile))
    labels[magnitude <= max(cutoff, 1e-8)] = np.nan
    return labels, cutoff


def shuffled_label_score(
    features,
    labels,
    train_idx,
    test_idx,
    ridge: float = 10.0,
    repeats: int = 20,
    seed: int = 0,
):
    """Return R² values after shuffling labels at the window level."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        shuffled = np.asarray(labels).copy()
        shuffled[train_idx] = shuffled[rng.permutation(train_idx)]
        truth, prediction, _ = fit_probe(features, shuffled, train_idx, test_idx, ridge)
        values.append(r2_score(truth, prediction))
    return np.asarray(values)


def residualize_against_position(labels, position_context, train_idx, ridge: float = 10.0):
    """Subtract the component predictable from XY using training windows only."""
    labels = np.asarray(labels)
    position_context = np.asarray(position_context)
    x_train, y_train = _flatten_valid(position_context, labels, train_idx)
    all_idx = np.arange(len(labels))
    x_all = position_context.reshape(-1, position_context.shape[-1])
    finite_all = np.isfinite(x_all).all(axis=-1)
    prediction = np.full((len(x_all),) + labels.shape[2:], np.nan, dtype=float)
    pred_valid, _ = standardize_fit(x_train, y_train, x_all[finite_all], ridge)
    if prediction.ndim == 2 and prediction.shape[1] == 1 and pred_valid.ndim == 1:
        pred_valid = pred_valid[:, None]
    prediction[finite_all] = pred_valid
    prediction = prediction.reshape(labels.shape)
    return labels - prediction


def bootstrap_metric_ci(truth, prediction, metric="r2", repeats=1000, seed=0):
    """Bootstrap rows of a held-out prediction and return a percentile interval."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(truth), len(truth))
        if metric == "r2":
            values.append(r2_score(truth[index], prediction[index]))
        elif metric == "cosine":
            values.append(direction_scores(truth[index], prediction[index])["cosine"])
        else:
            raise ValueError(f"unsupported metric {metric!r}")
    return np.quantile(values, [0.025, 0.975]).tolist()


def readability_onset(rows, value_key, control_key, consecutive=2, fraction_of_peak=0.5):
    """Find the first layer above control and a fraction of the family peak."""
    ordered = sorted(rows, key=lambda row: int(row["layer"]))
    peak = max(float(row[value_key]) for row in ordered)
    qualifies = [
        float(row[value_key]) > max(float(row[control_key]), 0.0)
        and float(row[value_key]) >= fraction_of_peak * peak
        for row in ordered
    ]
    for index in range(0, len(ordered) - consecutive + 1):
        if all(qualifies[index : index + consecutive]):
            return int(ordered[index]["layer"])
    return None


def save_activation_cache(path, representations, states, actions, choices, metadata=None):
    """Save pooled activation arrays with a name map in one portable NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name_map = {f"rep_{index:03d}": name for index, name in enumerate(sorted(representations))}
    payload = {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "choices": np.asarray(choices, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata or {})),
        "name_map_json": np.asarray(json.dumps(name_map)),
    }
    for key, name in name_map.items():
        payload[key] = np.asarray(representations[name])
    np.savez_compressed(path, **payload)


def load_activation_cache(path):
    with np.load(Path(path), allow_pickle=False) as payload:
        name_map = json.loads(str(payload["name_map_json"].item()))
        representations = {name: payload[key] for key, name in name_map.items()}
        metadata = json.loads(str(payload["metadata_json"].item()))
        return (
            representations,
            payload["states"],
            payload["actions"],
            payload["choices"],
            metadata,
        )
