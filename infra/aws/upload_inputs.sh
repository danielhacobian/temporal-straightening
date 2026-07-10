#!/usr/bin/env bash
set -Eeuo pipefail

REGION="${AWS_REGION:-us-east-2}"
STACK_NAME="${STACK_NAME:-temporal-straightening-infra}"

usage() {
  echo "Usage: $0 PATH_TO_POINT_MAZE_ZIP PATH_TO_FIXED_GOALS_PKL" >&2
  exit 2
}

[[ "$#" -eq 2 ]] || usage
dataset_path="$1"
goals_path="$2"
[[ -f "${dataset_path}" ]] || { echo "Dataset not found: ${dataset_path}" >&2; exit 2; }
[[ -f "${goals_path}" ]] || { echo "Goal bundle not found: ${goals_path}" >&2; exit 2; }

case "${dataset_path}" in
  *.zip) ;;
  *) echo "The canonical released dataset must be supplied as a .zip archive" >&2; exit 2 ;;
esac
case "${goals_path}" in
  *.pkl) ;;
  *) echo "The fixed goal bundle must be supplied as a .pkl file" >&2; exit 2 ;;
esac

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

bucket="$(
  aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='ArtifactBucketName'].OutputValue | [0]" \
    --output text
)"
[[ -n "${bucket}" && "${bucket}" != "None" ]] || {
  echo "Stack ${STACK_NAME} has no ArtifactBucketName output" >&2
  exit 2
}

upload_once() {
  local source="$1"
  local key="$2"
  local digest
  digest="$(sha256_file "${source}")"

  if existing="$(
    aws s3api head-object \
      --region "${REGION}" \
      --bucket "${bucket}" \
      --key "${key}" \
      --query 'Metadata.sha256' \
      --output text 2>/dev/null
  )"; then
    if [[ "${existing}" == "${digest}" ]]; then
      echo "Already present and verified: s3://${bucket}/${key} (${digest})"
      return
    fi
    echo "Refusing to overwrite canonical input s3://${bucket}/${key}" >&2
    echo "Use a new versioned key and update the broker/template intentionally." >&2
    exit 1
  fi

  aws s3 cp \
    --region "${REGION}" \
    --only-show-errors \
    --metadata "sha256=${digest}" \
    "${source}" "s3://${bucket}/${key}"

  observed="$(
    aws s3api head-object \
      --region "${REGION}" \
      --bucket "${bucket}" \
      --key "${key}" \
      --query 'Metadata.sha256' \
      --output text
  )"
  [[ "${observed}" == "${digest}" ]] || {
    echo "Uploaded metadata digest did not round-trip for ${key}" >&2
    exit 1
  }
  echo "Uploaded and verified: s3://${bucket}/${key} (${digest})"
}

upload_once "${dataset_path}" datasets/point_maze_umaze.zip
upload_once "${goals_path}" goals/umaze_fixed_v1.pkl
