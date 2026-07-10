#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/template.yaml"

export PYTHONDONTWRITEBYTECODE=1

for script in \
  "${SCRIPT_DIR}/deploy.sh" \
  "${SCRIPT_DIR}/upload_inputs.sh" \
  "${SCRIPT_DIR}/validate.sh" \
  "${REPO_ROOT}/infra/container/batch_entrypoint.sh"; do
  bash -n "${script}"
done

python3 -m unittest discover \
  --start-directory "${REPO_ROOT}/infra/tests" \
  --pattern 'test_*.py' \
  --verbose

(
  cd "${REPO_ROOT}"
  python3 -m infra.experiments.custom_manifest validate \
    infra/experiments/manifests/custom/*.yaml
)

if command -v cfn-lint >/dev/null 2>&1; then
  cfn-lint --regions us-east-2 --template "${TEMPLATE}"
elif python3 -c 'import cfnlint' >/dev/null 2>&1; then
  python3 -m cfnlint --regions us-east-2 --template "${TEMPLATE}"
elif [[ "${REQUIRE_CFN_LINT:-0}" == "1" ]]; then
  echo "cfn-lint is required but is not installed" >&2
  exit 1
else
  echo "warning: cfn-lint is not installed; structural tests still passed" >&2
fi

if [[ "${AWS_VALIDATE:-0}" == "1" ]]; then
  command -v aws >/dev/null 2>&1 || {
    echo "AWS_VALIDATE=1 requires AWS CLI v2" >&2
    exit 1
  }
  aws cloudformation validate-template \
    --region us-east-2 \
    --template-body "file://${TEMPLATE}" \
    >/dev/null
fi

echo "Infrastructure validation passed"
