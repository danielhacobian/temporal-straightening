import json
import math
import subprocess
import sys
from pathlib import Path


def write_seed(root: Path, seed: int, success: float, state: float) -> None:
    path = root / f"seed_{seed}" / "aggregate.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "seed": seed,
                "n_evals": 50,
                "metrics": {
                    "final_eval": {
                        "final_eval/success_rate": success,
                        "final_eval/mean_state_dist": state,
                    }
                },
            }
        )
        + "\n"
    )


def test_aggregate_condition_seeds(tmp_path: Path) -> None:
    off, on = tmp_path / "off", tmp_path / "on"
    for seed, off_success, on_success in ((100, 0.4, 0.5), (200, 0.6, 0.9)):
        write_seed(off, seed, off_success, 4.0)
        write_seed(on, seed, on_success, 3.0)

    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("aggregate_condition_seeds.py")),
            "--condition",
            f"off={off}",
            "--condition",
            f"on={on}",
            "--baseline",
            "off",
            "--treatment",
            "on",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    success = result["conditions"]["on"]["aggregate"]["final_eval/success_rate"]
    delta = result["paired_delta"]["metrics"]["final_eval/success_rate"]
    assert success == {"mean": 0.7, "population_std": 0.2}
    assert math.isclose(delta["mean"], 0.2)
    assert math.isclose(delta["population_std"], 0.1)

