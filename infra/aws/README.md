# AWS Batch infrastructure

This stack runs the paper-reproduction and screening funnel in `us-east-2`
without paying for idle compute. GPU jobs use one managed EC2 Spot compute
environment restricted to `g6.2xlarge` (one NVIDIA L4), and small proxy jobs can
use the optional Fargate Spot CPU queue.

## What the stack creates

- Two public subnets and an internet gateway, with no NAT gateway.
- An outbound-only security group with no inbound rules or SSH access.
- An S3 gateway endpoint, a private versioned artifact bucket, and lifecycle
  rules for scratch data, completed jobs, old versions, and incomplete uploads.
- An immutable, scan-on-push ECR repository.
- A zero-idle GPU Batch environment: Spot, `g6.2xlarge` only,
  `ECS_AL2023_NVIDIA`, min/desired 0 vCPU, max 8 vCPU, a 40% On-Demand bid
  ceiling, and `SPOT_PRICE_CAPACITY_OPTIMIZED` allocation.
- A GPU job definition requesting 1 GPU, 8 vCPU, and 28,672 MiB, with a 24-hour
  default hard timeout and retries only for infrastructure failures.
- An optional zero-idle Fargate Spot CPU queue for proxy analysis.
- A 14-day CloudWatch log group and narrowly scoped task, human observer,
  GitHub submit, and GitHub deployment roles. CI deployment and job submission
  deliberately use different roles.
- GitHub OIDC deploy trust limited to
  `danielhacobian/temporal-straightening` on `refs/heads/infra`; submission
  trust is limited to that branch, the five predefined `train-*` families,
  cost-bounded `train-chunk-*` requests, and constrained declarative
  `train-run-*` requests used by the CI broker.
- An optional automatic action on the already-existing net-spend budget that
  attaches an explicit deny to every normal launch/deployment role at 100%
  actual spend.

The broker reference is intentionally the mutable `infra` branch. Therefore
every repository writer who may change that branch is part of the AWS billing
trust boundary, even though normal contributors need no AWS identity. The
closed tag schema protects against accidental free-form input; it does not make
mutable workflow maintainers untrusted. Review changes under
`.github/workflows/`, `infra/aws/`, and `infra/experiments/` as
billing-sensitive changes, or pin the broker to an immutable commit if that
trust model changes.

There are deliberately no Organizations, IAM Identity Center, NAT gateway, SSH
key, or new AWS Budget resources in this template. The optional
`AWS::Budgets::BudgetsAction` references a budget that must already exist.

## Prerequisites

1. AWS CLI v2 authenticated to the intended account.
2. Docker with `buildx` (required because the Batch worker is `linux/amd64`).
3. Python 3 with PyYAML for local structural tests.
4. `cfn-lint` for full local validation.
5. Initial credentials that can create the named IAM, VPC, Batch, S3, ECR, and
   CloudWatch resources. Subsequent GitHub deployments use the narrower roles
   created by the bootstrap stack.

Install the linter in an isolated environment if needed:

```bash
python3 -m venv /tmp/ts-cfn-lint
/tmp/ts-cfn-lint/bin/pip install cfn-lint
PATH="/tmp/ts-cfn-lint/bin:$PATH" REQUIRE_CFN_LINT=1 ./infra/aws/validate.sh
```

Set `AWS_VALIDATE=1` to add AWS's authenticated `validate-template` call.

## Bootstrap and deploy

The deployment script validates the files, creates or updates the stack, builds
the image, and pushes it to ECR:

```bash
AWS_PROFILE=your-admin-profile ./infra/aws/deploy.sh
```

Defaults:

- Region: `us-east-2` (other regions are rejected)
- Stack: `temporal-straightening-infra`
- CPU Fargate queue: enabled
- GPU per-attempt timeout: 86,400 seconds (24 hours); manifests impose tighter
  per-run dollar and runtime ceilings for smoke and screening jobs.
- Image tag: deterministic `git-<commit>-<build-context-hash>`
- Existing-budget action: disabled until explicitly enabled

Useful overrides:

```bash
AWS_PROFILE=your-profile \
IMAGE_TAG="$GITHUB_SHA" \
ENABLE_CPU_FARGATE=true \
ENABLE_BUDGET_ACTION=true \
BUDGET_NAME=ts-net-out-of-pocket-20 \
BUDGET_EMAIL='owner@example.edu' \
GPU_JOB_TIMEOUT_SECONDS=86400 \
./infra/aws/deploy.sh
```

Enable the action only after confirming that `BUDGET_NAME` exactly matches the
existing **net out-of-pocket** cost budget. The action uses 100% `ACTUAL` spend
and `AUTOMATIC` approval. It blocks new Batch submissions and GitHub stack/image
deployments through the normal project roles. It does not create or change the
budget amount.

`IMAGE_TAG` values are immutable. Re-running an identical build reuses the
existing digest; changing code requires a new tag. The image has OCI source and
revision labels and prewarms DINOv2 ViT-S/14 from the pinned upstream commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`.

If `token.actions.githubusercontent.com` is already registered in the account,
the script passes that provider ARN into the stack. Otherwise the bootstrap
stack creates it. An account can have only one provider for that URL.

To deploy only CloudFormation and leave image publication to CI:

```bash
SKIP_IMAGE=true ./infra/aws/deploy.sh
```

After bootstrap, store `GitHubActionsRoleArn` in repository variable
`AWS_GITHUB_ROLE_ARN` and `GitHubSubmitRoleArn` in repository variable
`AWS_GITHUB_SUBMIT_ROLE_ARN`. Also set `AWS_BUDGET_ALERT_EMAIL` to the existing
Budget alert recipient; the template and deploy script intentionally contain no
personal default. The first role is owner-only and can deploy the
one stack and push its image but cannot submit jobs. The second can submit
through the broker but cannot edit infrastructure. Workflows request
`id-token: write`; there are no stored AWS keys and no protected-environment
reviewer. Manual dispatch is restricted
to the exact `infra` branch; the deploy role additionally requires the
`Deploy AWS infrastructure` workflow and the immutable GitHub actor ID for
`usharma123`. An owner push to `infra` invokes that deploy workflow with safe
defaults: build the commit-addressed image, keep the zero-idle CPU queue, and
retain the existing-budget action. A non-owner push skips the deployment job
and cannot assume the deploy role. Automatic submission is restricted to
`train-smoke-*`, `train-anchor-*`, `train-screen-*`, `train-finalists-*`, and
`train-scale-*`, plus cost-bounded `train-chunk-*` and declarative
`train-run-*` requests; the tiny trigger calls a reusable broker pinned to
`.github/workflows/aws-paid-broker.yml@refs/heads/infra`. That broker maps the
prefixes to fixed manifests and ceilings and requires the tag to target the
current `infra` tip. `train-smoke-*`, validated custom requests, and validated
single-run chunks may submit; the four costly full built-in mappings emit plans.
Pull-request and other branch/tag subjects cannot assume either role.

The actor-ID condition authenticates who starts deployment; it does not prove
who authored every file already at the branch tip. Because the next owner push
deploys the full current checkout, all `infra` writers remain an intentional
indirect infrastructure/image trust boundary. Review accumulated workflow,
template, and container changes before the owner pushes again.

`train-run-<slug>-<nonce>` selects only
`infra/experiments/manifests/custom/<slug>.yaml`. The broker validates its
closed schema before assuming AWS credentials and refuses unknown fields,
free-form commands/overrides, more than one approved variant, more than three
seeds, or a two-attempt aggregate envelope above $5. Compute resources,
canonical S3 versions, image, and entrypoint remain broker/stack controlled.

`train-chunk-<family>-<variant>-s<index>-r<rollouts>-<nonce>` selects exactly
one variant, one seed, and one rollout count from an allowlisted built-in
manifest. Families are `anchor`, `screen`, `finalists`, and `scale`; values
outside the checked-in manifest are rejected. The compiler preserves the
original run/checkpoint identity but clamps each attempt to $2.50, so the two
fixed infrastructure attempts reserve at most $5 total. The nonce is audit text
only: the same plan and source revision deduplicate even when the nonce changes.
An interrupted unit restores its uploaded epoch checkpoint; a genuinely new
run requires a reviewed source or request change.

Before submission, the broker acquires a versioned S3 compare-and-swap lease,
reconciles the budget ledger, and conditionally writes its reservation.
Admission requires `AWS Budget actual + every reserved/running or settling
worst case + the requested retry-inclusive maximum <= $15`. The other $5 of the
$20 Budget is never admitted to jobs; it remains a safety reserve for delayed
billing and small account charges. Separate tag
workflows are never collapsed by GitHub concurrency; budgeted jobs may queue in
AWS Batch and the one-worker compute ceiling runs them sequentially. A failed
submit is recovered by deterministic Batch name/tags or its reservation remains
held. Only a later workflow, after a 15-minute grace and a second no-job proof,
may conditionally expire an abandoned reservation.
Terminal reservations remain charged for seven days so delayed AWS Budgets data
cannot immediately reopen the same spend.

Contributors who use the GitHub workflow need only repository push permission.
They do not need an AWS identity or to share an email address. For example:

```bash
git switch infra
git pull --ff-only origin infra
git tag train-smoke-alice-ci1
git push origin train-smoke-alice-ci1
```

For smoke, custom, chunk, and full-matrix plan tags, the suffix is an audit label
only. The same deterministic plan and source revision deduplicate regardless of
nonce; a genuinely new run needs a reviewed source or request change. The tag
still cannot provide a manifest, profile, command, price, or budget override.
Pushing `train-smoke-*` queues the mapped
job after deterministic-plan, cumulative-budget admission, one-use
idempotency, and live-price checks. Full anchor/screen/finalist/scale prefixes
generate plans only; their explicit chunk form can submit one bounded run.
Deleting a tag does not stop an existing Batch job. A GitHub Actions
display/run name alone never authorizes a submission.

## Give a teammate observer access (optional)

The stack output `LauncherRoleArn` is assumable by identities in the same AWS
account, but the trust policy alone does not grant access. An administrator must
give each teammate's existing IAM identity this one permission:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "sts:AssumeRole",
    "Resource": "<LauncherRoleArn>"
  }]
}
```

Then configure a local profile (no long-lived project access key is needed):

```ini
[profile temporal-straightening]
role_arn = <LauncherRoleArn>
source_profile = default
region = us-east-2
```

The role is now an observer: it can inspect and stop project jobs and read job
logs/artifacts, but it cannot submit work or write S3 objects. Normal
contributors do not need this AWS role at all; repository push permission is
enough to use the pinned GitHub broker. Keeping raw `batch:SubmitJob` away from
human identities prevents bypassing the broker's timeout, price, and
idempotency checks.

## Upload the released dataset once

The paid broker reads the two canonical URIs from stack outputs. From an
owner/admin session, use the helper to upload the released archive and extended
fixed-goal file once with their SHA-256 in S3 metadata:

```bash
./infra/aws/upload_inputs.sh point_maze_umaze.zip umaze_fixed_v1.pkl
```

The submit role can read exactly those two keys but cannot overwrite them. It
requires a version ID and 64-hex `sha256` metadata before planning a paid run.
Workers independently checksum the downloaded bytes and can write only to
`jobs/` and `scratch/`, so training cannot overwrite canonical inputs.

## Economical smoke tests

An account owner can test the container and S3/log plumbing on CPU first from
the bootstrap session; this direct helper path is not granted to contributors:

```bash
python3 infra/aws/submit_job.py \
  submit --compute cpu --job-name ts-cpu-smoke --attempts 1
```

Then test GPU visibility and the prewarmed DINO model. The default command is
still only the fast smoke test; it does not start training:

```bash
python3 infra/aws/submit_job.py \
  submit --compute gpu --job-name ts-gpu-smoke \
  --max-runtime-seconds 1800 --attempts 1 --max-compute-usd 1
```

GPU submission queries the current `g6.2xlarge` Spot prices in `us-east-2` and
rejects the job when the estimated maximum runtime times retry attempts exceeds
`--max-compute-usd`. If pricing cannot be retrieved, the helper fails closed;
`--allow-unpriced` is an explicit escape hatch.

## Submit the bounded UMaze smoke through CI

Contributors use the broker, not the owner-only helper:

```bash
git switch infra
git pull --ff-only origin infra
git tag train-smoke-alice-ci1
git push origin train-smoke-alice-ci1
```

The broker accepts the tag only at the current `infra` tip, resolves the fixed
manifest and canonical versioned inputs, enforces the $0.80 two-attempt plan
ceiling, checks Spot price, conditionally reserves cumulative budget, and queues
the Batch job. Full anchor, screen, finalist, and scale tags remain plan-only.

Submit one screened unit, for example seed index 0 of the normalized-
acceleration variant at the manifest's fixed 100 rollouts:

```bash
git tag train-chunk-screen-r2_normalized_acceleration-s0-r100-utsav01
git push origin train-chunk-screen-r2_normalized_acceleration-s0-r100-utsav01
```

Other examples are
`train-chunk-anchor-projector_only-s0-r2000-<nonce>`,
`train-chunk-finalists-paper_curvature-s1-r200-<nonce>`, and
`train-chunk-scale-projector_only-s0-r800-<nonce>`. Invalid variant, seed, and
rollout selectors fail before AWS authentication.

## Observe and stop jobs

```bash
python3 infra/aws/submit_job.py --profile temporal-straightening status JOB_ID
python3 infra/aws/submit_job.py --profile temporal-straightening logs JOB_ID --wait-seconds 300
python3 infra/aws/submit_job.py --profile temporal-straightening stop JOB_ID
```

`stop` uses `CancelJob` while a job is queued and `TerminateJob` after it starts.
AWS Batch does not retry cancelled or terminated jobs.

## Cost boundaries and limitations

- Idle GPU and CPU compute is zero because min/desired capacity is zero.
- The 8-vCPU GPU ceiling permits only one `g6.2xlarge` worker at a time.
- The job definition has a hard per-attempt timeout; the broker's manifest
  bounds each attempt and accounts for both infrastructure attempts.
- Fargate/EC2 public IPv4, EBS, S3, ECR, and CloudWatch can still incur small
  non-GPU charges. There is no NAT gateway hourly charge.
- The already-created console budget remains outside CloudFormation, avoiding a
  duplicate budget. When explicitly enabled, the stack adds only an automatic
  action to that existing budget; at 100% actual spend it applies an explicit
  deny to the human observer, GitHub submit, and GitHub deployment roles so
  they cannot start more work or replace the stack through the normal paths.
- AWS Budgets uses delayed billing data, so the broker supplements it with a
  conditional S3 ledger and admits only when actual spend plus all outstanding
  worst-case reservations plus the new request is at most $15, leaving $5 of
  the $20 Budget permanently unallocated as a safety reserve. The seven-day
  terminal hold is deliberately conservative and may delay new work while
  credits/billing settle. The budget action does not terminate an already-
  running job or block an account administrator, so keep alerts enabled and use
  `stop` when necessary.
- Spot estimates can change after submission. The hard runtime and one-instance
  ceiling bound exposure, but they cannot guarantee an exact final invoice.

The artifact bucket and ECR repository are retained if the stack is deleted so
research evidence cannot disappear with an infrastructure teardown. Remove
them manually only after exporting anything that must be preserved.
