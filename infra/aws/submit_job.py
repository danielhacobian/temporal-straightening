#!/usr/bin/env python3
"""Submit, inspect, and stop project jobs through the AWS CLI.

The submit path applies a hard per-attempt runtime and, for GPU jobs, rejects
the submission when the current Spot-price estimate exceeds the caller's
per-job ceiling. This is an execution guard, not an account-wide billing cap.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


REGION = "us-east-2"
DEFAULT_STACK = "temporal-straightening-infra"
DEFAULT_GPU_RUNTIME_SECONDS = 43_200
DEFAULT_CPU_RUNTIME_SECONDS = 7_200
MAX_GPU_RUNTIME_SECONDS = 86_400
MAX_CPU_RUNTIME_SECONDS = 14_400
DEFAULT_GPU_JOB_CEILING_USD = 10.0
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class AwsCli:
    def __init__(self, profile: str | None) -> None:
        self.base = ["aws", "--region", REGION]
        if profile:
            self.base.extend(["--profile", profile])

    def json(self, *args: str) -> Any:
        completed = subprocess.run(
            [*self.base, *args, "--output", "json"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return json.loads(completed.stdout)

    def call(self, *args: str) -> None:
        subprocess.run([*self.base, *args], check=True)


def stack_outputs(aws: AwsCli, stack_name: str) -> dict[str, str]:
    response = aws.json("cloudformation", "describe-stacks", "--stack-name", stack_name)
    outputs = response["Stacks"][0].get("Outputs", [])
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


def current_gpu_spot_hourly_usd(aws: AwsCli) -> float:
    now = datetime.now(timezone.utc).isoformat()
    response = aws.json(
        "ec2",
        "describe-spot-price-history",
        "--instance-types",
        "g6.2xlarge",
        "--product-descriptions",
        "Linux/UNIX",
        "--start-time",
        now,
        "--max-results",
        "20",
    )
    prices = [float(item["SpotPrice"]) for item in response.get("SpotPriceHistory", [])]
    if not prices:
        raise RuntimeError("AWS returned no current g6.2xlarge Spot prices in us-east-2")
    return max(prices)


def parse_env(items: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Environment override must be KEY=VALUE: {item!r}")
        name, value = item.split("=", 1)
        if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        result.append({"name": name, "value": value})
    return result


def submit(aws: AwsCli, args: argparse.Namespace) -> None:
    outputs = stack_outputs(aws, args.stack_name)
    project_name = outputs.get("ProjectName", "temporal-straightening")
    cpu = args.compute == "cpu"
    queue_key = "CpuJobQueueArn" if cpu else "GpuJobQueueArn"
    definition_key = "CpuJobDefinitionArn" if cpu else "GpuJobDefinitionArn"
    if queue_key not in outputs or definition_key not in outputs:
        raise RuntimeError(f"{args.compute} queue is not enabled in stack {args.stack_name}")

    default_runtime = DEFAULT_CPU_RUNTIME_SECONDS if cpu else DEFAULT_GPU_RUNTIME_SECONDS
    max_runtime = MAX_CPU_RUNTIME_SECONDS if cpu else MAX_GPU_RUNTIME_SECONDS
    runtime_seconds = args.max_runtime_seconds or default_runtime
    if not 60 <= runtime_seconds <= max_runtime:
        raise ValueError(
            f"--max-runtime-seconds must be between 60 and {max_runtime} for {args.compute} jobs"
        )

    estimate: dict[str, float] | None = None
    if args.max_compute_usd <= 0:
        raise ValueError("--max-compute-usd must be positive")
    if not cpu:
        try:
            hourly = current_gpu_spot_hourly_usd(aws)
        except (RuntimeError, subprocess.CalledProcessError):
            if not args.allow_unpriced:
                raise RuntimeError(
                    "Could not price the GPU submission. Re-authenticate or pass "
                    "--allow-unpriced only after checking the EC2 Spot console."
                ) from None
        else:
            estimated_usd = hourly * (runtime_seconds / 3600) * args.attempts
            estimate = {
                "current_spot_hourly_usd": round(hourly, 6),
                "worst_case_attempt_runtime_usd": round(estimated_usd, 2),
                "submission_ceiling_usd": args.max_compute_usd,
            }
            if estimated_usd > args.max_compute_usd:
                raise RuntimeError(
                    f"Estimated worst-case compute ${estimated_usd:.2f} exceeds "
                    f"--max-compute-usd ${args.max_compute_usd:.2f}. Reduce runtime/attempts."
                )

    command = args.command or ["python", "infra/container/smoke_test.py"]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after --")
    container_overrides: dict[str, Any] = {"command": command}
    environment = parse_env(args.env)
    if environment:
        container_overrides["environment"] = environment

    retry_rules = [
        {"action": "RETRY", "onReason": "AGENT"},
        {"action": "RETRY", "onStatusReason": "Task failed to start*"},
        {"action": "EXIT", "onReason": "*"},
    ]
    if not cpu:
        retry_rules.insert(
            0, {"action": "RETRY", "onStatusReason": "Host EC2*"}
        )

    payload = aws.json(
        "batch",
        "submit-job",
        "--job-name",
        args.job_name,
        "--job-queue",
        outputs[queue_key],
        "--job-definition",
        outputs[definition_key],
        "--container-overrides",
        json.dumps(container_overrides, separators=(",", ":")),
        "--timeout",
        json.dumps({"attemptDurationSeconds": runtime_seconds}),
        "--retry-strategy",
        json.dumps(
            {"attempts": args.attempts, "evaluateOnExit": retry_rules},
            separators=(",", ":"),
        ),
        "--tags",
        json.dumps(
            {
                "Project": project_name,
                "SubmittedBy": os.environ.get("USER", "unknown")[:128],
            },
            separators=(",", ":"),
        ),
    )
    print(
        json.dumps(
            {
                "job_id": payload["jobId"],
                "job_name": payload["jobName"],
                "compute": args.compute,
                "max_runtime_seconds_per_attempt": runtime_seconds,
                "attempts": args.attempts,
                "spot_estimate": estimate,
            },
            indent=2,
        )
    )


def describe_job(aws: AwsCli, job_id: str) -> dict[str, Any]:
    response = aws.json("batch", "describe-jobs", "--jobs", job_id)
    if not response.get("jobs"):
        raise RuntimeError(f"Job not found: {job_id}")
    return response["jobs"][0]


def status(aws: AwsCli, args: argparse.Namespace) -> None:
    job = describe_job(aws, args.job_id)
    selected = {
        key: job.get(key)
        for key in (
            "jobId",
            "jobName",
            "status",
            "statusReason",
            "createdAt",
            "startedAt",
            "stoppedAt",
            "attempts",
            "container",
        )
    }
    print(json.dumps(selected, indent=2))


def stop(aws: AwsCli, args: argparse.Namespace) -> None:
    job = describe_job(aws, args.job_id)
    state = job["status"]
    if state in {"SUBMITTED", "PENDING", "RUNNABLE"}:
        aws.call("batch", "cancel-job", "--job-id", args.job_id, "--reason", args.reason)
        action = "cancelled"
    elif state in {"STARTING", "RUNNING"}:
        aws.call("batch", "terminate-job", "--job-id", args.job_id, "--reason", args.reason)
        action = "terminated"
    else:
        action = f"unchanged ({state})"
    print(json.dumps({"job_id": args.job_id, "action": action}, indent=2))


def logs(aws: AwsCli, args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.wait_seconds
    stream = None
    while time.monotonic() <= deadline:
        job = describe_job(aws, args.job_id)
        stream = job.get("container", {}).get("logStreamName")
        if stream:
            break
        if args.wait_seconds == 0:
            break
        time.sleep(5)
    if not stream:
        raise RuntimeError("The job has no CloudWatch log stream yet")
    outputs = stack_outputs(aws, args.stack_name)
    response = aws.json(
        "logs",
        "get-log-events",
        "--log-group-name",
        outputs["BatchLogGroupName"],
        "--log-stream-name",
        stream,
        "--start-from-head",
    )
    for event in response.get("events", []):
        print(event["message"])


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--profile", help="AWS CLI profile, normally the launcher role profile")
    root.add_argument("--stack-name", default=DEFAULT_STACK)
    root.add_argument("--region", default=REGION, choices=[REGION])
    subparsers = root.add_subparsers(dest="action", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a bounded CPU or GPU job")
    submit_parser.add_argument("--compute", choices=["cpu", "gpu"], default="gpu")
    submit_parser.add_argument("--job-name", required=True, type=valid_job_name)
    submit_parser.add_argument("--max-runtime-seconds", type=int)
    submit_parser.add_argument("--attempts", type=int, choices=range(1, 4), default=2)
    submit_parser.add_argument("--max-compute-usd", type=float, default=DEFAULT_GPU_JOB_CEILING_USD)
    submit_parser.add_argument("--allow-unpriced", action="store_true")
    submit_parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    submit_parser.add_argument("command", nargs=argparse.REMAINDER)
    submit_parser.set_defaults(handler=submit)

    status_parser = subparsers.add_parser("status", help="Describe one job")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(handler=status)

    stop_parser = subparsers.add_parser("stop", help="Cancel a queued job or terminate a running job")
    stop_parser.add_argument("job_id")
    stop_parser.add_argument("--reason", default="Stopped by project launcher")
    stop_parser.set_defaults(handler=stop)

    logs_parser = subparsers.add_parser("logs", help="Print the current CloudWatch log stream")
    logs_parser.add_argument("job_id")
    logs_parser.add_argument("--wait-seconds", type=int, default=0)
    logs_parser.set_defaults(handler=logs)
    return root


def valid_job_name(value: str) -> str:
    if not JOB_NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("Use 1-128 letters, digits, underscores, or hyphens")
    return value


def main() -> None:
    args = parser().parse_args()
    try:
        args.handler(AwsCli(args.profile), args)
    except (KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
