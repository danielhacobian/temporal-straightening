#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TEMPLATE="${SCRIPT_DIR}/template.yaml"

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-2}}"
STACK_NAME="${STACK_NAME:-temporal-straightening-infra}"
PROJECT_NAME="${PROJECT_NAME:-temporal-straightening}"
ENABLE_CPU_FARGATE="${ENABLE_CPU_FARGATE:-true}"
ENABLE_BUDGET_ACTION="${ENABLE_BUDGET_ACTION:-false}"
BUDGET_NAME="${BUDGET_NAME:-ts-net-out-of-pocket-20}"
BUDGET_EMAIL="${BUDGET_EMAIL:-}"
GPU_JOB_TIMEOUT_SECONDS="${GPU_JOB_TIMEOUT_SECONDS:-86400}"
SKIP_IMAGE="${SKIP_IMAGE:-false}"

if [[ "${REGION}" != "us-east-2" ]]; then
  echo "This stack is restricted to us-east-2; got ${REGION}" >&2
  exit 2
fi
if [[ "${ENABLE_CPU_FARGATE}" != "true" && "${ENABLE_CPU_FARGATE}" != "false" ]]; then
  echo "ENABLE_CPU_FARGATE must be true or false" >&2
  exit 2
fi
if [[ "${ENABLE_BUDGET_ACTION}" != "true" && "${ENABLE_BUDGET_ACTION}" != "false" ]]; then
  echo "ENABLE_BUDGET_ACTION must be true or false" >&2
  exit 2
fi
if [[ ! "${BUDGET_EMAIL}" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]]; then
  echo "Set BUDGET_EMAIL to the existing budget alert recipient." >&2
  exit 2
fi

command -v aws >/dev/null 2>&1 || {
  echo "AWS CLI v2 is required" >&2
  exit 1
}

"${SCRIPT_DIR}/validate.sh"

ACCOUNT_ID="$(aws sts get-caller-identity --region "${REGION}" --query Account --output text)"
if [[ -z "${ACCOUNT_ID}" || "${ACCOUNT_ID}" == "None" ]]; then
  echo "Unable to determine the active AWS account" >&2
  exit 1
fi

SOURCE_SHA="${GITHUB_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
if [[ -z "${IMAGE_TAG:-}" ]]; then
  CONTEXT_HASH="$(REPO_ROOT="${REPO_ROOT}" python3 - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
top_level_python = list(root.glob("*.py"))
trees = [
    root / "conf",
    root / "datasets",
    root / "distributed_fn",
    root / "env",
    root / "metrics",
    root / "models",
    root / "planning",
    root / "infra" / "container",
    root / "infra" / "experiments",
]
paths = top_level_python
for tree in trees:
    if tree.exists():
        paths.extend(path for path in tree.rglob("*") if path.is_file())

def excluded(path: Path) -> bool:
    relative = path.relative_to(root)
    text = relative.as_posix()
    return (
        "__pycache__" in relative.parts
        or text.startswith("env/deformable_env/src/sim/assets/")
        or text.startswith("models/encoder/r3m/")
        or path.suffix == ".pyc"
    )

digest = hashlib.sha256()
for path in sorted((path for path in paths if not excluded(path)), key=lambda item: item.as_posix()):
    relative = path.relative_to(root).as_posix().encode()
    digest.update(len(relative).to_bytes(4, "big"))
    digest.update(relative)
    digest.update(path.read_bytes())
print(digest.hexdigest()[:16])
PY
)"
  IMAGE_TAG="git-${SOURCE_SHA:0:12}-${CONTEXT_HASH}"
fi

# Preserve ownership across updates. If this stack created the provider, its
# parameter is empty and must remain empty; switching to the discovered ARN
# would make CloudFormation delete the provider before reusing that same ARN.
if aws cloudformation describe-stacks \
  --region "${REGION}" \
  --stack-name "${STACK_NAME}" \
  >/dev/null 2>&1; then
  OIDC_PROVIDER_ARN="$(
    aws cloudformation describe-stacks \
      --region "${REGION}" \
      --stack-name "${STACK_NAME}" \
      --query "Stacks[0].Parameters[?ParameterKey=='GitHubOidcProviderArn'].ParameterValue | [0]" \
      --output text
  )"
else
  OIDC_PROVIDER_ARN="$(
    aws iam list-open-id-connect-providers \
      --query "OpenIDConnectProviderList[?ends_with(Arn, 'oidc-provider/token.actions.githubusercontent.com')].Arn | [0]" \
      --output text
  )"
fi
if [[ "${OIDC_PROVIDER_ARN}" == "None" ]]; then
  OIDC_PROVIDER_ARN=""
fi

CF_EXECUTION_ROLE="$(
  aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFormationExecutionRoleArn'].OutputValue | [0]" \
    --output text 2>/dev/null || true
)"
if [[ "${CF_EXECUTION_ROLE}" == "None" ]]; then
  CF_EXECUTION_ROLE=""
fi

parameter_overrides=(
  "ProjectName=${PROJECT_NAME}"
  "ContainerImageTag=${IMAGE_TAG}"
  "EnableCpuFargate=${ENABLE_CPU_FARGATE}"
  "EnableExistingBudgetAction=${ENABLE_BUDGET_ACTION}"
  "ExistingBudgetName=${BUDGET_NAME}"
  "BudgetAlertEmail=${BUDGET_EMAIL}"
  "GpuJobTimeoutSeconds=${GPU_JOB_TIMEOUT_SECONDS}"
)
if [[ -n "${OIDC_PROVIDER_ARN}" ]]; then
  parameter_overrides+=("GitHubOidcProviderArn=${OIDC_PROVIDER_ARN}")
fi

deploy_args=(
  cloudformation deploy
  --region "${REGION}"
  --stack-name "${STACK_NAME}"
  --template-file "${TEMPLATE}"
  --capabilities CAPABILITY_NAMED_IAM
  --no-fail-on-empty-changeset
  --parameter-overrides
    "${parameter_overrides[@]}"
  --tags
    "Project=${PROJECT_NAME}"
    "ManagedBy=CloudFormation"
)
if [[ -n "${CF_EXECUTION_ROLE}" ]]; then
  deploy_args+=(--role-arn "${CF_EXECUTION_ROLE}")
fi

echo "Deploying ${STACK_NAME} in account ${ACCOUNT_ID}, region ${REGION}"
aws "${deploy_args[@]}"

REPOSITORY_URI="$(
  aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='TrainingRepositoryUri'].OutputValue | [0]" \
    --output text
)"

if [[ "${SKIP_IMAGE}" != "true" ]]; then
  command -v docker >/dev/null 2>&1 || {
    echo "Docker with buildx is required unless SKIP_IMAGE=true" >&2
    exit 1
  }
  docker buildx version >/dev/null

  REPOSITORY_NAME="${PROJECT_NAME}/training"
  EXISTING_DIGEST="$(
    aws ecr describe-images \
      --region "${REGION}" \
      --repository-name "${REPOSITORY_NAME}" \
      --image-ids "imageTag=${IMAGE_TAG}" \
      --query 'imageDetails[0].imageDigest' \
      --output text 2>/dev/null || true
  )"
  if [[ -n "${EXISTING_DIGEST}" && "${EXISTING_DIGEST}" != "None" ]]; then
    echo "Immutable image ${REPOSITORY_URI}:${IMAGE_TAG} already exists (${EXISTING_DIGEST}); skipping push"
  else
    aws ecr get-login-password --region "${REGION}" \
      | docker login \
          --username AWS \
          --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    docker buildx build \
      --platform linux/amd64 \
      --file "${REPO_ROOT}/infra/container/Dockerfile" \
      --build-arg "SOURCE_REVISION=${SOURCE_SHA}" \
      --tag "${REPOSITORY_URI}:${IMAGE_TAG}" \
      --push \
      "${REPO_ROOT}"
  fi
fi

echo
echo "Deployment complete"
echo "  stack: ${STACK_NAME}"
echo "  image: ${REPOSITORY_URI}:${IMAGE_TAG}"
echo "  outputs: aws cloudformation describe-stacks --region ${REGION} --stack-name ${STACK_NAME} --query 'Stacks[0].Outputs'"
