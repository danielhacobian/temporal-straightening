"""Cross-file security contracts for the trusted AWS submission broker."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from infra.experiments.broker_ledger import BROKER_WORKFLOW_TIMEOUT, LOCK_LEASE


REPO_ROOT = Path(__file__).resolve().parents[2]
BROKER_PATH = REPO_ROOT / ".github" / "workflows" / "aws-paid-broker.yml"
REQUEST_PATH = REPO_ROOT / ".github" / "workflows" / "aws-submit.yml"
TEMPLATE_PATH = REPO_ROOT / "infra" / "aws" / "template.yaml"
RUNNER_PATH = REPO_ROOT / "infra" / "experiments" / "runner.py"
CUSTOM_COMPILER_PATH = REPO_ROOT / "infra" / "experiments" / "custom_manifest.py"
CHUNK_COMPILER_PATH = REPO_ROOT / "infra" / "experiments" / "builtin_chunk.py"
LEDGER_PATH = REPO_ROOT / "infra" / "experiments" / "broker_ledger.py"

REPOSITORY = "danielhacobian/temporal-straightening"
REPOSITORY_ID = "1278580902"
OWNER_ID = "126291787"
DEPLOY_ACTOR_ID = "125384326"
DEPLOY_WORKFLOW = "Deploy AWS infrastructure"
BROKER_WORKFLOW_REF = (
    "danielhacobian/temporal-straightening/"
    ".github/workflows/aws-paid-broker.yml@refs/heads/infra"
)
TAG_FAMILIES = {
    "train-smoke-*": ("smoke", "default", "submit"),
    "train-anchor-*": ("umaze_exact_anchor", "minimal", "plan-only"),
    "train-screen-*": ("screening_funnel", "default", "plan-only"),
    "train-finalists-*": ("finalists", "default", "plan-only"),
    "train-scale-*": ("scaling_trend", "default", "plan-only"),
}
CUSTOM_TAG_FAMILY = "train-run-*"
CHUNK_TAG_FAMILY = "train-chunk-*"
ALL_TAG_FAMILIES = [*TAG_FAMILIES, CHUNK_TAG_FAMILY, CUSTOM_TAG_FAMILY]


class StrictLoader(yaml.SafeLoader):
    """Load YAML/CloudFormation short tags while rejecting duplicate keys."""


def strict_mapping(
    loader: StrictLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def cloudformation_tag(
    loader: StrictLoader, tag_suffix: str, node: yaml.Node
) -> dict[str, Any]:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {f"Fn::{tag_suffix}": value}


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, strict_mapping
)
StrictLoader.add_multi_constructor("!", cloudformation_tag)


def load_yaml(path: Path) -> dict[Any, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)


def allowed_actions(policy: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for statement in policy["Statement"]:
        if statement.get("Effect") != "Allow":
            continue
        action = statement.get("Action", [])
        if isinstance(action, str):
            result.add(action)
        else:
            result.update(action)
    return result


class TrustedBrokerWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.broker = load_yaml(BROKER_PATH)
        cls.request = load_yaml(REQUEST_PATH)
        cls.template = load_yaml(TEMPLATE_PATH)
        cls.resources = cls.template["Resources"]
        cls.broker_job = cls.broker["jobs"]["broker"]
        cls.request_job = cls.request["jobs"]["broker"]
        cls.steps = cls.broker_job["steps"]
        cls.steps_by_name = {step["name"]: step for step in cls.steps}

    def case_block(self, tag_pattern: str) -> str:
        run = self.steps_by_name["Resolve a fail-closed trigger policy"]["run"]
        match = re.search(
            rf"(?ms)^\s*{re.escape(tag_pattern)}\)\s*$\n(?P<body>.*?)(?=^\s*;;\s*$)",
            run,
        )
        self.assertIsNotNone(match, f"missing case block for {tag_pattern}")
        return match.group("body")

    def test_no_protected_environment_changes_the_oidc_subject(self) -> None:
        for workflow in (self.broker, self.request):
            for job in workflow["jobs"].values():
                self.assertNotIn("environment", job)

        encoded_template = json.dumps(self.template)
        self.assertNotIn("token.actions.githubusercontent.com:environment", encoded_template)
        self.assertNotIn("environment:aws-paid", encoded_template)
        self.assertNotIn("concurrency", self.broker)
        self.assertNotIn("concurrency", self.request)

    def test_smoke_submits_and_every_costly_family_is_plan_only(self) -> None:
        trigger_tags = self.request[True]["push"]["tags"]
        self.assertEqual(trigger_tags, ALL_TAG_FAMILIES)

        resolve_run = self.steps_by_name["Resolve a fail-closed trigger policy"]["run"]
        # Smoke, constrained custom requests, and one-run built-in chunks may
        # submit. The four full confirmatory matrices remain plan-only.
        self.assertEqual(resolve_run.count("mode=submit"), 3)
        for tag, (manifest, profile, mode) in TAG_FAMILIES.items():
            block = self.case_block(tag)
            self.assertIn(f"manifest={manifest}", block)
            self.assertIn(f"profile={profile}", block)
            self.assertIn(f"mode={mode}", block)

        custom_start = resolve_run.index('if [[ "${REF_NAME}" =~ ^train-run-')
        custom_end = resolve_run.index('elif [[ "${REF_NAME}" =~ ^train-', custom_start)
        custom_block = resolve_run[custom_start:custom_end]
        self.assertIn(
            'custom_spec="infra/experiments/manifests/custom/${custom_name}.yaml"',
            custom_block,
        )
        self.assertIn('manifest="custom_${custom_name}"', custom_block)
        self.assertIn("maximum_plan_usd=5.00", custom_block)
        self.assertIn("mode=submit", custom_block)

        chunk_start = resolve_run.index('if [[ "${REF_NAME}" =~ ^train-chunk-')
        chunk_end = resolve_run.index('elif [[ "${REF_NAME}" =~ ^train-run-', chunk_start)
        chunk_block = resolve_run[chunk_start:chunk_end]
        self.assertIn("chunk_family=", chunk_block)
        self.assertIn("chunk_variant=", chunk_block)
        self.assertIn("chunk_seed_index=", chunk_block)
        self.assertIn("chunk_rollouts=", chunk_block)
        self.assertIn("maximum_plan_usd=5.00", chunk_block)
        self.assertIn("mode=submit", chunk_block)

        manual = re.search(
            r'(?ms)elif \[\[ "\$\{EVENT_NAME\}" == "workflow_dispatch" \]\]; then'
            r"(?P<body>.*?)(?=^\s*else\s*$)",
            resolve_run,
        )
        self.assertIsNotNone(manual)
        self.assertIn("mode=plan-only", manual.group("body"))

        submit_step = self.steps_by_name["Submit the cost-bounded GPU array"]
        self.assertIn("mode == 'submit'", submit_step["if"])
        self.assertIn("steps.ledger.outputs.should_submit == 'true'", submit_step["if"])
        self.assertEqual(
            sum("aws batch submit-job" in step.get("run", "") for step in self.steps),
            1,
        )

    def test_live_price_and_budget_are_checked_before_submit(self) -> None:
        names = [step["name"] for step in self.steps]
        contract_name = "Verify the fixed one-GPU execution contract"
        price_name = "Recheck the current L4 Spot rate immediately before submission"
        lock_name = "Recover idempotency state and expire stranded legacy locks"
        ledger_name = "Reserve aggregate budget with the S3 admission lease"
        submit_name = "Submit the cost-bounded GPU array"
        self.assertLess(names.index(price_name), names.index(lock_name))
        self.assertLess(names.index(lock_name), names.index(ledger_name))
        self.assertLess(names.index(ledger_name), names.index(submit_name))
        self.assertLess(names.index(lock_name), names.index(submit_name))

        contract = self.steps_by_name[contract_name]["run"]
        self.assertIn("aws batch describe-job-queues", contract)
        self.assertIn("aws batch describe-compute-environments", contract)
        self.assertIn('"instanceTypes": ["g6.2xlarge"]', contract)
        self.assertIn('"bidPercentage": 40', contract)
        self.assertIn('"maxvCpus": 8', contract)
        self.assertIn('"minvCpus": 0', contract)

        price = self.steps_by_name[price_name]["run"]
        self.assertIn("--instance-types g6.2xlarge", price)
        self.assertNotIn("g6.4xlarge", price)
        submit_environment = self.steps_by_name[submit_name]["env"]
        self.assertEqual(
            submit_environment["HOURLY_USD"],
            "${{ steps.resolve.outputs.maximum_hourly_usd }}",
        )
        self.assertNotEqual(
            submit_environment["HOURLY_USD"],
            "${{ steps.price.outputs.hourly_usd }}",
        )
        summary = self.steps_by_name["Summarize the broker decision"]
        self.assertEqual(
            summary["env"]["USABLE_ADMISSION_USD"],
            "${{ steps.ledger.outputs.usable_admission_usd }}",
        )
        self.assertIn("Billing safety reserve", summary["run"])

        ledger = self.steps_by_name[ledger_name]
        self.assertIn("steps.resolve.outputs.mode == 'submit'", ledger["if"])
        self.assertIn("steps.idempotency.outputs.should_submit == 'true'", ledger["if"])
        self.assertIn("infra.experiments.broker_ledger reserve", ledger["run"])
        self.assertIn("--budget-name ts-net-out-of-pocket-20", ledger["run"])
        self.assertIn('--maximum-total-usd "${MAXIMUM_WITH_RETRIES_USD}"', ledger["run"])

        submit_index = names.index(submit_name)
        for step in self.steps[:submit_index]:
            self.assertNotIn("aws batch submit-job", step.get("run", ""))

    def test_budget_ledger_is_conditional_recoverable_and_retry_inclusive(self) -> None:
        ledger = LEDGER_PATH.read_text(encoding="utf-8")
        self.assertIn('BUDGET_LIMIT_USD = Decimal("20.00")', ledger)
        self.assertIn('SAFETY_RESERVE_USD = Decimal("5.00")', ledger)
        self.assertIn(
            "ADMISSION_LIMIT_USD = BUDGET_LIMIT_USD - SAFETY_RESERVE_USD",
            ledger,
        )
        self.assertIn('MAX_REQUEST_USD = Decimal("5.00")', ledger)
        self.assertIn("actual_usd + outstanding_usd + requested_usd", ledger)
        self.assertIn('"--show-filter-expression"', ledger)
        self.assertIn('OUTSTANDING_STATES = {"reserved", "submitted", "settling"}', ledger)
        self.assertIn('arguments.extend(("--if-none-match", "*"))', ledger)
        self.assertIn('arguments.extend(("--if-match", expected_etag))', ledger)
        self.assertLess(
            ledger.index("job = store.find_job(entry)"),
            ledger.index('"abandoned-reservation-with-no-batch-job"'),
        )
        self.assertIn("SETTLEMENT_HOLD = timedelta(days=7)", ledger)
        self.assertIn('f"name=JOB_NAME,values={entry[\'job_name\']}"', ledger)
        self.assertIn('ADMISSION_LOCK_KEY = "broker-locks/budget-admission.json"', ledger)
        self.assertIn("with held_admission_lease(store, owner=owner) as lease:", ledger)
        self.assertIn("assert_admission_lease_current(store, lease)", ledger)
        self.assertNotIn("def first_active_job", ledger)
        self.assertEqual(
            self.broker_job["timeout-minutes"],
            int(BROKER_WORKFLOW_TIMEOUT.total_seconds() // 60),
        )
        self.assertGreater(LOCK_LEASE, BROKER_WORKFLOW_TIMEOUT)
        for status in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"):
            self.assertIn(status, ledger)

        names = [step["name"] for step in self.steps]
        submit = names.index("Submit the cost-bounded GPU array")
        finalize = names.index("Finalize or safely release the budget reservation")
        marker = names.index("Record the immutable submission marker")
        self.assertLess(submit, finalize)
        self.assertLess(finalize, marker)
        finalizer = self.steps_by_name[
            "Finalize or safely release the budget reservation"
        ]
        self.assertIn("always()", finalizer["if"])
        self.assertIn("infra.experiments.broker_ledger finalize", finalizer["run"])

        statements = self.resources["GitHubBrokerManagedPolicy"]["Properties"][
            "PolicyDocument"
        ]["Statement"]
        list_ledger = next(item for item in statements if item["Sid"] == "ListBudgetLedger")
        maintain = next(item for item in statements if item["Sid"] == "MaintainBudgetLedger")
        list_lock = next(item for item in statements if item["Sid"] == "ListAdmissionLock")
        maintain_lock = next(
            item for item in statements if item["Sid"] == "MaintainAdmissionLock"
        )
        self.assertEqual(list_ledger["Action"], "s3:ListBucket")
        self.assertIn("budget-ledger/*", json.dumps(list_ledger))
        self.assertEqual(set(maintain["Action"]), {"s3:GetObject", "s3:PutObject"})
        self.assertIn("${ArtifactBucket.Arn}/budget-ledger/*", json.dumps(maintain))
        self.assertEqual(list_lock["Action"], "s3:ListBucket")
        self.assertIn("broker-locks/budget-admission.json", json.dumps(list_lock))
        self.assertEqual(
            set(maintain_lock["Action"]), {"s3:GetObject", "s3:PutObject"}
        )
        bucket_policy = self.resources["ArtifactBucketPolicy"]["Properties"][
            "PolicyDocument"
        ]["Statement"]
        conditional = next(
            item
            for item in bucket_policy
            if item["Sid"] == "RequireConditionalBrokerStateWrites"
        )
        self.assertEqual(conditional["Effect"], "Deny")
        self.assertEqual(
            conditional["Condition"]["Null"],
            {"s3:if-match": "true", "s3:if-none-match": "true"},
        )

    def test_builtin_chunk_is_one_static_run_with_a_five_dollar_ceiling(self) -> None:
        validation_name = "Validate the built-in chunk before AWS authentication"
        plan_name = "Build the deterministic array plan"
        submit_name = "Submit the cost-bounded GPU array"
        validation = self.steps_by_name[validation_name]["run"]
        plan = self.steps_by_name[plan_name]["run"]
        submit = self.steps_by_name[submit_name]["run"]
        self.assertIn("infra.experiments.builtin_chunk compile", validation)
        self.assertIn("infra.experiments.builtin_chunk plan", plan)
        self.assertIn('"infra.experiments.builtin_chunk", "run"', submit)
        compiler = CHUNK_COMPILER_PATH.read_text(encoding="utf-8")
        self.assertIn("MAX_TOTAL_USD = 5.0", compiler)
        self.assertIn("len(runs) == 1", compiler)
        self.assertIn("chunk[\"seed_sets\"] = [", compiler)
        self.assertIn("chunk[\"variants\"] = [", compiler)

    def test_canonical_s3_hashes_and_version_ids_reach_the_worker(self) -> None:
        stack = self.steps_by_name["Resolve immutable stack outputs"]["run"]
        self.assertIn("s3://${bucket}/datasets/point_maze_umaze.zip", stack)
        self.assertIn("s3://${bucket}/goals/umaze_fixed_v1.pkl", stack)

        inputs = self.steps_by_name["Pin canonical input metadata"]["run"]
        self.assertIn("--key datasets/point_maze_umaze.zip", inputs)
        self.assertIn("--key goals/umaze_fixed_v1.pkl", inputs)
        self.assertIn('get("Metadata", {}).get("sha256"', inputs)
        self.assertIn('head.get("VersionId")', inputs)
        self.assertIn("_version_id={version_id}", inputs)

        expected_env = {
            "TS_UMAZE_DATASET_S3_URI": "${{ steps.inputs.outputs.dataset_uri }}",
            "TS_UMAZE_DATASET_SHA256": "${{ steps.inputs.outputs.dataset_sha256 }}",
            "TS_UMAZE_DATASET_VERSION_ID": "${{ steps.inputs.outputs.dataset_version_id }}",
            "TS_UMAZE_GOALS_S3_URI": "${{ steps.inputs.outputs.goal_set_uri }}",
            "TS_UMAZE_GOALS_SHA256": "${{ steps.inputs.outputs.goals_sha256 }}",
            "TS_UMAZE_GOALS_VERSION_ID": "${{ steps.inputs.outputs.goals_version_id }}",
        }
        for name in ("Build the deterministic array plan", "Submit the cost-bounded GPU array"):
            environment = self.steps_by_name[name]["env"]
            for key, value in expected_env.items():
                self.assertEqual(environment[key], value)

        submit_run = self.steps_by_name["Submit the cost-bounded GPU array"]["run"]
        for key in expected_env:
            self.assertIn(f'"{key}"', submit_run)
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertRegex(runner, r'"--version-id",\s+version_id')
        self.assertIn("run.dataset_version_id,", runner)
        self.assertIn("run.goal_set_version_id,", runner)

        outputs = self.template["Outputs"]
        self.assertEqual(
            outputs["UmazeDatasetUri"]["Value"],
            {"Fn::Sub": "s3://${ArtifactBucket}/datasets/point_maze_umaze.zip"},
        )
        self.assertEqual(
            outputs["UmazeGoalSetUri"]["Value"],
            {"Fn::Sub": "s3://${ArtifactBucket}/goals/umaze_fixed_v1.pkl"},
        )

    def test_oidc_claims_pin_repo_ids_refs_and_reusable_workflow(self) -> None:
        deploy_condition = self.resources["GitHubActionsRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]["Statement"][0]["Condition"]
        self.assertEqual(
            deploy_condition["StringEquals"],
            {
                "token.actions.githubusercontent.com:actor_id": DEPLOY_ACTOR_ID,
                "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                "token.actions.githubusercontent.com:ref": "refs/heads/infra",
                "token.actions.githubusercontent.com:repository": REPOSITORY,
                "token.actions.githubusercontent.com:repository_id": REPOSITORY_ID,
                "token.actions.githubusercontent.com:repository_owner_id": OWNER_ID,
                "token.actions.githubusercontent.com:sub": (
                    f"repo:{REPOSITORY}:ref:refs/heads/infra"
                ),
                "token.actions.githubusercontent.com:workflow": DEPLOY_WORKFLOW,
            },
        )

        submit_condition = self.resources["GitHubSubmitRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]["Statement"][0]["Condition"]
        self.assertEqual(
            submit_condition["StringEquals"],
            {
                "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                "token.actions.githubusercontent.com:job_workflow_ref": BROKER_WORKFLOW_REF,
                "token.actions.githubusercontent.com:repository": REPOSITORY,
                "token.actions.githubusercontent.com:repository_id": REPOSITORY_ID,
                "token.actions.githubusercontent.com:repository_owner_id": OWNER_ID,
            },
        )
        expected_refs = [
            "refs/heads/infra",
            *[f"refs/tags/{tag}" for tag in ALL_TAG_FAMILIES],
        ]
        self.assertEqual(
            submit_condition["StringLike"]["token.actions.githubusercontent.com:ref"],
            expected_refs,
        )
        self.assertEqual(
            submit_condition["StringLike"]["token.actions.githubusercontent.com:sub"],
            [f"repo:{REPOSITORY}:ref:{ref}" for ref in expected_refs],
        )
        self.assertEqual(
            self.request_job["uses"],
            f"{REPOSITORY}/.github/workflows/aws-paid-broker.yml@infra",
        )

    def test_custom_requests_are_declarative_fixed_gpu_jobs_under_five_dollars(
        self,
    ) -> None:
        names = [step["name"] for step in self.steps]
        validation_name = "Validate the declarative custom request before AWS authentication"
        contract_name = "Verify the fixed one-GPU execution contract"
        plan_name = "Build the deterministic array plan"
        submit_name = "Submit the cost-bounded GPU array"
        assume_index = next(
            index
            for index, step in enumerate(self.steps)
            if str(step.get("uses", "")).startswith(
                "aws-actions/configure-aws-credentials@"
            )
        )
        self.assertLess(names.index(validation_name), assume_index)
        self.assertLess(names.index(contract_name), names.index(plan_name))
        self.assertLess(names.index(plan_name), names.index(submit_name))

        validation = self.steps_by_name[validation_name]["run"]
        self.assertIn("infra.experiments.custom_manifest compile", validation)
        plan = self.steps_by_name[plan_name]["run"]
        self.assertIn("infra.experiments.custom_manifest plan", plan)
        self.assertIn('--budget-usd "${MAXIMUM_PLAN_USD}"', plan)
        submit = self.steps_by_name[submit_name]["run"]
        self.assertIn('"infra.experiments.custom_manifest", "run"', submit)

        contract = self.steps_by_name[contract_name]["run"]
        self.assertIn('resources.get("GPU") != "1"', contract)
        self.assertIn('get("attempts") != 2', contract)
        self.assertIn("timeout > 86400", contract)
        self.assertIn('platformCapabilities") != ["EC2"]', contract)

        compiler = CUSTOM_COMPILER_PATH.read_text(encoding="utf-8")
        self.assertIn("MAX_TOTAL_USD = 5.0", compiler)
        self.assertIn('"uri": "${TS_UMAZE_DATASET_S3_URI}"', compiler)
        self.assertIn('"version_id": "${TS_UMAZE_DATASET_VERSION_ID}"', compiler)
        self.assertIn('"uri": "${TS_UMAZE_GOALS_S3_URI}"', compiler)
        self.assertIn('"version_id": "${TS_UMAZE_GOALS_VERSION_ID}"', compiler)

        gpu = self.resources["GpuJobDefinition"]["Properties"]
        resources = {
            item["Type"]: item["Value"]
            for item in gpu["ContainerProperties"]["ResourceRequirements"]
        }
        self.assertEqual(resources["GPU"], "1")
        self.assertEqual(gpu["PlatformCapabilities"], ["EC2"])
        self.assertEqual(gpu["RetryStrategy"]["Attempts"], 2)
        self.assertEqual(
            gpu["Timeout"]["AttemptDurationSeconds"],
            {"Fn::Ref": "GpuJobTimeoutSeconds"},
        )
        timeout = self.template["Parameters"]["GpuJobTimeoutSeconds"]
        self.assertLessEqual(timeout["MaxValue"], 86_400)

        submit_statement = next(
            item
            for item in self.resources["GitHubBrokerManagedPolicy"]["Properties"]
            ["PolicyDocument"]["Statement"]
            if item["Sid"] == "SubmitToProjectQueues"
        )
        self.assertEqual(
            submit_statement["Resource"],
            [
                {
                    "Fn::Sub": (
                        "arn:${AWS::Partition}:batch:${AWS::Region}:"
                        "${AWS::AccountId}:job/*"
                    )
                },
                {"Fn::Ref": "GpuJobQueue"},
                {"Fn::Ref": "GpuJobDefinition"},
            ],
        )

    def test_observer_is_read_stop_only_and_cannot_write_or_submit(self) -> None:
        observer = self.resources["BatchObserverManagedPolicy"]["Properties"]
        policy = observer["PolicyDocument"]
        actions = allowed_actions(policy)
        self.assertEqual(observer["Roles"], [{"Fn::Ref": "BatchLauncherRole"}])
        self.assertTrue({"batch:CancelJob", "batch:TerminateJob"} <= actions)
        for forbidden in (
            "batch:SubmitJob",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:AbortMultipartUpload",
            "cloudformation:UpdateStack",
            "ecr:PutImage",
        ):
            self.assertNotIn(forbidden, actions)

        encoded = json.dumps(policy)
        self.assertIn("${ArtifactBucket.Arn}/jobs/*", encoded)
        self.assertNotIn("${ArtifactBucket.Arn}/datasets/*", encoded)
        self.assertNotIn("${ArtifactBucket.Arn}/goals/*", encoded)

        broker = self.resources["GitHubBrokerManagedPolicy"]["Properties"]
        self.assertEqual(broker["Roles"], [{"Fn::Ref": "GitHubSubmitRole"}])
        self.assertIn("batch:SubmitJob", allowed_actions(broker["PolicyDocument"]))
        deploy_role = self.resources["GitHubActionsRole"]["Properties"]
        self.assertNotIn("batch:SubmitJob", json.dumps(deploy_role.get("Policies", [])))
        self.assertNotIn("Policies", self.resources["GitHubSubmitRole"]["Properties"])

    def test_cloudformation_policies_match_workflow_calls(self) -> None:
        deploy_policy = self.resources["GitHubActionsRole"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]
        deploy_actions = allowed_actions(deploy_policy)
        self.assertIn("cloudformation:ValidateTemplate", deploy_actions)
        self.assertIn("cloudformation:DescribeStacks", deploy_actions)
        self.assertIn("ecr:BatchGetImage", deploy_actions)
        self.assertIn("ecr:DescribeImages", deploy_actions)
        execution_policy = self.resources["CloudFormationExecutionRole"]["Properties"][
            "Policies"
        ][0]["PolicyDocument"]
        execution_actions = allowed_actions(execution_policy)
        self.assertIn("ecr:GetRepositoryPolicy", execution_actions)
        self.assertIn("iam:ListOpenIDConnectProviderTags", execution_actions)

        broker_policy = self.resources["GitHubBrokerManagedPolicy"]["Properties"][
            "PolicyDocument"
        ]
        broker_actions = allowed_actions(broker_policy)
        self.assertIn("cloudformation:DescribeStacks", broker_actions)
        self.assertIn("budgets:ViewBudget", broker_actions)
        self.assertIn("s3:GetObjectVersion", broker_actions)

        job_statements = self.resources["JobRole"]["Properties"]["Policies"][0][
            "PolicyDocument"
        ]["Statement"]
        location = next(item for item in job_statements if item["Sid"] == "LocateArtifactBucket")
        listing = next(item for item in job_statements if item["Sid"] == "InspectArtifactBucket")
        self.assertNotIn("Condition", location)
        self.assertIn("Condition", listing)
        self.assertLess(TEMPLATE_PATH.stat().st_size, 51_200)


if __name__ == "__main__":
    unittest.main()
