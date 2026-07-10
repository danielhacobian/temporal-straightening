"""Measure held-out latent goal-distance monotonicity without planner compute."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import deque
from pathlib import Path
from statistics import median
from typing import Sequence


MAZES = {
    "umaze": (
        "#####",
        "#GOO#",
        "###O#",
        "#OOO#",
        "#####",
    )
}


def average_ranks(values: Sequence[float]) -> list[float]:
    """Assign 1-based average ranks, including ties."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = rank
        cursor = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman's rho with average tie ranks and no SciPy dependency."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    return numerator / denominator


def _open_cells(maze: Sequence[str]) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row, line in enumerate(maze)
        for column, value in enumerate(line)
        if value != "#"
    ]


def nearest_cell(position: Sequence[float], maze: Sequence[str]) -> tuple[int, int]:
    """Map continuous MuJoCo coordinates to the nearest navigable cell center."""
    x, y = float(position[0]), float(position[1])
    cells = _open_cells(maze)
    if not cells:
        raise ValueError("maze has no navigable cells")
    return min(cells, key=lambda cell: (cell[0] - x) ** 2 + (cell[1] - y) ** 2)


def shortest_grid_steps(
    start: Sequence[float], goal: Sequence[float], maze: Sequence[str]
) -> int:
    """Return the unweighted A*/BFS shortest cell path between two states."""
    source, target = nearest_cell(start, maze), nearest_cell(goal, maze)
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        (row, column), distance = queue.popleft()
        if (row, column) == target:
            return distance
        for candidate in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            r, c = candidate
            if (
                0 <= r < len(maze)
                and 0 <= c < len(maze[r])
                and maze[r][c] != "#"
                and candidate not in seen
            ):
                seen.add(candidate)
                queue.append((candidate, distance + 1))
    raise ValueError(f"no path between {source} and {target}")


def summarize_group_rhos(groups: Sequence[dict]) -> dict[str, float | int]:
    """Aggregate correlations only after ranking states within each goal group."""
    if not groups:
        raise ValueError("at least one trajectory/goal group is required")
    rhos = [
        spearman(group["latent_distances"], group["shortest_path_steps"])
        for group in groups
    ]
    return {
        "groups": len(rhos),
        "spearman_rho_mean": sum(rhos) / len(rhos),
        "spearman_rho_median": median(rhos),
        "positive_group_fraction": sum(value > 0 for value in rhos) / len(rhos),
    }


def _find_checkpoint(model_dir: Path, epoch: str) -> tuple[Path, int | str]:
    """Resolve an explicit epoch or the newest numeric checkpoint safely.

    Training may also write ``model_latest.pth``.  It matches the numeric glob,
    but its suffix cannot be converted to an integer, so keep it as a fallback
    for runs that do not retain numbered checkpoints.
    """
    if epoch != "latest":
        candidate = model_dir / "checkpoints" / f"model_{epoch}.pth"
        if candidate.is_file():
            return candidate, int(epoch)

    numbered = []
    for candidate in (model_dir / "checkpoints").glob("model_*.pth"):
        suffix = candidate.stem.removeprefix("model_")
        if suffix.isdigit():
            numbered.append((int(suffix), candidate))
    if numbered:
        checkpoint_epoch, candidate = max(numbered, key=lambda item: item[0])
        return candidate, checkpoint_epoch

    latest = model_dir / "checkpoints" / "model_latest.pth"
    if latest.is_file():
        return latest, "latest"
    raise FileNotFoundError(f"no model checkpoint under {model_dir}")


def evaluate(model_dir: Path, epoch: str, goal_set: Path, maze_name: str) -> dict:
    """Encode fixed goals and correlate latent distance with shortest-path steps."""
    import hydra
    import torch
    from omegaconf import OmegaConf

    from plan import load_model
    from preprocessor import Preprocessor

    if maze_name not in MAZES:
        raise ValueError(f"unsupported maze {maze_name}")
    with (model_dir / "hydra.yaml").open("r", encoding="utf-8") as handle:
        config = OmegaConf.load(handle)
    checkpoint, checkpoint_epoch = _find_checkpoint(model_dir, epoch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, config, config.num_action_repeat, device=device)
    model.eval()

    _, trajectory_datasets = hydra.utils.call(
        config.env.dataset,
        num_hist=config.num_hist,
        num_pred=config.num_pred,
        frameskip=config.frameskip,
    )
    dataset = trajectory_datasets["valid"].dataset
    preprocessor = Preprocessor(
        action_mean=dataset.action_mean,
        action_std=dataset.action_std,
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        proprio_mean=dataset.proprio_mean,
        proprio_std=dataset.proprio_std,
        transform=dataset.transform,
    )
    with goal_set.open("rb") as handle:
        targets = pickle.load(handle)
    trajectories = targets.get("proxy_trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(
            "goal set needs proxy_trajectories with multiple states per fixed goal; "
            "independent start/goal pairs cannot validate goal-distance monotonicity"
        )

    def transformed(obs):
        return {key: value.to(device) for key, value in preprocessor.transform_obs(obs).items()}

    group_results = []
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            raise ValueError(f"proxy_trajectories[{index}] must be a mapping")
        for key in ("observations", "goal_observation"):
            if key not in trajectory:
                raise ValueError(f"proxy_trajectories[{index}] is missing {key}")
        with torch.no_grad():
            state_latent = model.encode_obs(transformed(trajectory["observations"]))[
                "visual"
            ]
            goal_latent = model.encode_obs(transformed(trajectory["goal_observation"]))[
                "visual"
            ]
            if goal_latent.shape[0] != 1:
                raise ValueError(
                    f"proxy_trajectories[{index}].goal_observation must have batch size 1"
                )
            dimensions = tuple(range(1, state_latent.ndim))
            latent_distances = torch.linalg.vector_norm(
                state_latent - goal_latent, dim=dimensions
            ).detach().cpu().tolist()
        if "shortest_path_steps" in trajectory:
            path_steps = [float(value) for value in trajectory["shortest_path_steps"]]
            path_source = "precomputed_astar"
        else:
            states = trajectory.get("states")
            goal_state = trajectory.get("goal_state")
            if states is None or goal_state is None:
                raise ValueError(
                    f"proxy_trajectories[{index}] needs shortest_path_steps or states+goal_state"
                )
            path_steps = [
                float(shortest_grid_steps(state, goal_state, MAZES[maze_name]))
                for state in states
            ]
            path_source = f"grid_astar:{maze_name}"
        if len(latent_distances) != len(path_steps) or len(path_steps) < 3:
            raise ValueError(
                f"proxy_trajectories[{index}] needs >=3 aligned latent/path samples"
            )
        rho = spearman(latent_distances, path_steps)
        group_results.append(
            {
                "trajectory_id": str(trajectory.get("trajectory_id", index)),
                "goal_id": str(trajectory.get("goal_id", index)),
                "samples": len(path_steps),
                "spearman_rho": rho,
                "path_distance_source": path_source,
                "latent_distances": [float(value) for value in latent_distances],
                "shortest_path_steps": path_steps,
            }
        )
    summary = summarize_group_rhos(group_results)
    result = {
        "metric": "candidate_goal_distance_monotonicity_signal",
        "interpretation": "candidate screening signal; not validated by this computation alone",
        **summary,
        "model_dir": str(model_dir.resolve()),
        "model_checkpoint": str(checkpoint.resolve()),
        "model_epoch": checkpoint_epoch,
        "goal_set": str(goal_set.resolve()),
        "group_results": group_results,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--goal-set", type=Path, required=True)
    parser.add_argument("--maze", choices=sorted(MAZES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.model_dir, args.model_epoch, args.goal_set, args.maze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "group_results"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
