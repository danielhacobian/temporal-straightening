#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${RUN_ROOT:-/workspace/run}"
DATASET_DIR="${DATASET_DIR:-/workspace/data}"
APP_ROOT="${APP_ROOT:-/opt/temporal-straightening}"
JOB_ID="${AWS_BATCH_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_URI="${S3_ARTIFACT_URI:-}"

if [[ -z "${ARTIFACT_URI}" && -n "${ARTIFACT_BUCKET:-}" ]]; then
  ARTIFACT_URI="s3://${ARTIFACT_BUCKET}/jobs/${JOB_ID}/"
fi

mkdir -p "${RUN_ROOT}" "${DATASET_DIR}"

if [[ -n "${S3_INPUT_URI:-}" ]]; then
  echo "Syncing input data from ${S3_INPUT_URI} to ${DATASET_DIR}"
  aws s3 sync "${S3_INPUT_URI}" "${DATASET_DIR}" --only-show-errors
fi

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"
COMMAND_DISPLAY="$(printf '%q ' "$@")"

finalize() {
  local original_code=$?
  local final_code="${original_code}"
  trap - EXIT INT TERM
  set +e

  EXIT_CODE="${original_code}" \
  STARTED_AT="${STARTED_AT}" \
  STARTED_EPOCH="${STARTED_EPOCH}" \
  COMMAND_DISPLAY="${COMMAND_DISPLAY}" \
  JOB_ID="${JOB_ID}" \
  RUN_ROOT="${RUN_ROOT}" \
    python - <<'PY'
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

exit_code = int(os.environ["EXIT_CODE"])
payload = {
    "job_id": os.environ["JOB_ID"],
    "state": "succeeded" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "command": os.environ["COMMAND_DISPLAY"].strip(),
    "started_at": os.environ["STARTED_AT"],
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": max(0, time.time() - int(os.environ["STARTED_EPOCH"])),
}
Path(os.environ["RUN_ROOT"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["RUN_ROOT"]) / "batch_status.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
PY

  if [[ -n "${ARTIFACT_URI}" ]]; then
    echo "Syncing job artifacts to ${ARTIFACT_URI}"
    if ! aws s3 sync "${RUN_ROOT}" "${ARTIFACT_URI}" --only-show-errors; then
      echo "Artifact upload failed" >&2
      if [[ "${final_code}" -eq 0 ]]; then
        final_code=74
      fi
    fi
  fi

  exit "${final_code}"
}

child_pid=0
watchdog_pid=0
termination_grace_seconds="${TERMINATION_GRACE_SECONDS:-105}"
forward_signal() {
  local signal="$1"
  if [[ "${child_pid}" -gt 0 ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "Forwarding ${signal} to child process ${child_pid}; allowing ${termination_grace_seconds}s for checkpoint upload" >&2
    kill "-${signal}" "${child_pid}" 2>/dev/null || true
    if [[ "${watchdog_pid}" -eq 0 ]]; then
      (
        sleep "${termination_grace_seconds}"
        if kill -0 "${child_pid}" 2>/dev/null; then
          echo "Termination grace expired; killing child ${child_pid}" >&2
          kill -KILL "${child_pid}" 2>/dev/null || true
        fi
      ) &
      watchdog_pid=$!
    fi
  fi
}

trap finalize EXIT
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

cd "${APP_ROOT}"
echo "Running: ${COMMAND_DISPLAY}"
set +e
"$@" &
child_pid=$!
while true; do
  wait "${child_pid}"
  child_code=$?
  if ! kill -0 "${child_pid}" 2>/dev/null; then
    break
  fi
  # A signal interrupted wait; the child is still flushing its own finally block.
done
if [[ "${watchdog_pid}" -gt 0 ]]; then
  kill "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
fi
set -e
exit "${child_code}"
