"""Structural guardrails for the cost-bounded AWS infrastructure."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "infra" / "aws" / "template.yaml"
DOCKERFILE_PATH = REPO_ROOT / "infra" / "container" / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / "infra" / "container" / "Dockerfile.dockerignore"
DEPLOY_SCRIPT_PATH = REPO_ROOT / "infra" / "aws" / "deploy.sh"


class CloudFormationLoader(yaml.SafeLoader):
    """Parse short-form CloudFormation tags without evaluating them."""


def cloudformation_tag(
    loader: CloudFormationLoader, tag_suffix: str, node: yaml.Node
) -> dict[str, Any]:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {f"Fn::{tag_suffix}": value}


CloudFormationLoader.add_multi_constructor("!", cloudformation_tag)


class InfrastructureTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = TEMPLATE_PATH.read_text(encoding="utf-8")
        cls.template = yaml.load(cls.raw, Loader=CloudFormationLoader)
        cls.resources = cls.template["Resources"]

    def test_region_is_explicitly_restricted(self) -> None:
        self.assertEqual(self.template["Metadata"]["DeploymentRegion"], "us-east-2")
        self.assertIn("RequireUsEast2", self.template["Rules"])
        self.assertIn("us-east-2", self.raw)

    def test_public_network_has_no_nat_or_inbound_rules(self) -> None:
        resource_types = {item["Type"] for item in self.resources.values()}
        self.assertNotIn("AWS::EC2::NatGateway", resource_types)
        for name in ("PublicSubnetA", "PublicSubnetB"):
            self.assertIs(self.resources[name]["Properties"]["MapPublicIpOnLaunch"], True)
        security_group = self.resources["BatchSecurityGroup"]["Properties"]
        self.assertNotIn("SecurityGroupIngress", security_group)
        self.assertEqual(security_group["SecurityGroupEgress"][0]["FromPort"], 443)
        endpoint = self.resources["S3GatewayEndpoint"]["Properties"]
        self.assertEqual(endpoint["VpcEndpointType"], "Gateway")
        self.assertIn(".s3", json.dumps(endpoint["ServiceName"]))
        launch_template = self.resources["GpuLaunchTemplate"]["Properties"]
        self.assertEqual(
            launch_template["LaunchTemplateData"]["MetadataOptions"]["HttpPutResponseHopLimit"],
            1,
        )
        user_data = json.dumps(launch_template["LaunchTemplateData"]["UserData"])
        self.assertIn("ECS_CONTAINER_STOP_TIMEOUT=120s", user_data)
        self.assertIn("multipart/mixed", user_data)

    def test_gpu_compute_is_zero_idle_single_g6_spot_worker(self) -> None:
        compute = self.resources["GpuComputeEnvironment"]["Properties"]
        self.assertEqual(compute["Type"], "MANAGED")
        resources = compute["ComputeResources"]
        self.assertEqual(resources["Type"], "SPOT")
        self.assertEqual(resources["AllocationStrategy"], "SPOT_PRICE_CAPACITY_OPTIMIZED")
        self.assertEqual(resources["InstanceTypes"], ["g6.2xlarge"])
        self.assertEqual(resources["MinvCpus"], 0)
        self.assertEqual(resources["DesiredvCpus"], 0)
        self.assertEqual(resources["MaxvCpus"], 8)
        self.assertEqual(resources["BidPercentage"], 40)
        self.assertEqual(resources["Ec2Configuration"][0]["ImageType"], "ECS_AL2023_NVIDIA")

    def test_gpu_job_has_requested_resources_and_safety_controls(self) -> None:
        job = self.resources["GpuJobDefinition"]["Properties"]
        requirements = {
            item["Type"]: item["Value"]
            for item in job["ContainerProperties"]["ResourceRequirements"]
        }
        self.assertEqual(requirements, {"GPU": "1", "VCPU": "8", "MEMORY": "28672"})
        self.assertEqual(job["PlatformCapabilities"], ["EC2"])
        self.assertEqual(
            job["ContainerProperties"]["Command"],
            ["python", "-m", "infra.container.smoke_test"],
        )
        self.assertNotIn("RuntimePlatform", job["ContainerProperties"])
        self.assertEqual(job["RetryStrategy"]["Attempts"], 2)
        self.assertIn("EvaluateOnExit", job["RetryStrategy"])
        self.assertIn("AttemptDurationSeconds", job["Timeout"])
        self.assertFalse(job["ContainerProperties"]["Privileged"])

    def test_cpu_queue_is_optional_fargate_spot_with_public_ip(self) -> None:
        compute = self.resources["CpuComputeEnvironment"]
        self.assertEqual(compute["Condition"], "CreateCpuFargate")
        self.assertEqual(compute["Properties"]["ComputeResources"]["Type"], "FARGATE_SPOT")
        job = self.resources["CpuJobDefinition"]
        self.assertEqual(job["Condition"], "CreateCpuFargate")
        properties = job["Properties"]
        self.assertEqual(properties["PlatformCapabilities"], ["FARGATE"])
        self.assertEqual(
            properties["ContainerProperties"]["Command"],
            ["python", "-m", "infra.container.smoke_test"],
        )
        self.assertEqual(
            properties["ContainerProperties"]["NetworkConfiguration"]["AssignPublicIp"],
            "ENABLED",
        )

    def test_artifacts_are_private_versioned_and_lifecycle_managed(self) -> None:
        resource = self.resources["ArtifactBucket"]
        bucket = resource["Properties"]
        self.assertEqual(resource["DeletionPolicy"], "Retain")
        self.assertEqual(bucket["VersioningConfiguration"]["Status"], "Enabled")
        self.assertTrue(bucket["PublicAccessBlockConfiguration"]["RestrictPublicBuckets"])
        self.assertGreaterEqual(len(bucket["LifecycleConfiguration"]["Rules"]), 3)
        self.assertIn("BucketEncryption", bucket)
        bucket_policy = self.resources["ArtifactBucketPolicy"]["Properties"]["PolicyDocument"]
        self.assertIn("aws:SecureTransport", json.dumps(bucket_policy))

    def test_repository_is_immutable_and_logs_expire(self) -> None:
        repository = self.resources["TrainingRepository"]["Properties"]
        self.assertEqual(repository["ImageTagMutability"], "IMMUTABLE")
        self.assertTrue(repository["ImageScanningConfiguration"]["ScanOnPush"])
        self.assertEqual(
            self.resources["BatchLogGroup"]["Properties"]["RetentionInDays"], 14
        )

    def test_github_trust_is_exact_repo_branch_and_pinned_broker(self) -> None:
        deploy_trust = self.resources["GitHubActionsRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]
        deploy_encoded = json.dumps(deploy_trust)
        self.assertIn("sts.amazonaws.com", deploy_encoded)
        self.assertIn(
            "repo:danielhacobian/temporal-straightening:ref:refs/heads/infra",
            deploy_encoded,
        )
        self.assertIn('1278580902', deploy_encoded)
        self.assertNotIn("repo:danielhacobian/temporal-straightening:*", deploy_encoded)

        submit_trust = self.resources["GitHubSubmitRole"]["Properties"][
            "AssumeRolePolicyDocument"
        ]
        submit_encoded = json.dumps(submit_trust)
        self.assertIn(
            "danielhacobian/temporal-straightening/.github/workflows/aws-paid-broker.yml@refs/heads/infra",
            submit_encoded,
        )
        self.assertIn("refs/tags/train-smoke-*", submit_encoded)
        self.assertIn("refs/heads/infra", submit_encoded)
        self.assertNotIn("environment:aws-paid", submit_encoded)

    def test_observer_cannot_submit_or_write_and_broker_is_narrow(self) -> None:
        observer = self.resources["BatchObserverManagedPolicy"]["Properties"]
        observer_encoded = json.dumps(observer["PolicyDocument"])
        for action in (
            "batch:CancelJob",
            "batch:TerminateJob",
            "batch:DescribeJobs",
        ):
            self.assertIn(action, observer_encoded)
        self.assertNotIn("batch:SubmitJob", observer_encoded)
        self.assertNotIn("s3:PutObject", observer_encoded)
        self.assertEqual([{"Fn::Ref": "BatchLauncherRole"}], observer["Roles"])

        broker = self.resources["GitHubBrokerManagedPolicy"]["Properties"]
        broker_encoded = json.dumps(broker["PolicyDocument"])
        self.assertIn("batch:SubmitJob", broker_encoded)
        self.assertIn("budgets:ViewBudget", broker_encoded)
        self.assertIn("/submissions/*", broker_encoded)
        broker_resources = json.dumps(
            [
                statement.get("Resource")
                for statement in broker["PolicyDocument"]["Statement"]
            ]
        )
        self.assertNotIn("${ArtifactBucket.Arn}/*", broker_resources)
        self.assertEqual([{"Fn::Ref": "GitHubSubmitRole"}], broker["Roles"])

    def test_jobs_cannot_overwrite_pinned_inputs(self) -> None:
        job_policy = self.resources["JobRole"]["Properties"]["Policies"][0]
        encoded = json.dumps(job_policy)
        self.assertIn("/datasets/*", encoded)
        self.assertIn("/goals/*", encoded)
        self.assertIn("/jobs/*", encoded)
        self.assertNotIn("s3:DeleteObject", encoded)

    def test_stack_does_not_create_org_identity_center_or_a_second_budget(self) -> None:
        forbidden_fragments = ("Organizations", "SSO", "IdentityCenter")
        types = [item["Type"] for item in self.resources.values()]
        for resource_type in types:
            self.assertFalse(
                any(fragment in resource_type for fragment in forbidden_fragments),
                resource_type,
            )
        self.assertNotIn("AWS::Budgets::Budget", types)

    def test_existing_budget_action_is_optional_and_blocks_new_launches(self) -> None:
        action = self.resources["ExistingBudgetGuardAction"]
        self.assertEqual(action["Condition"], "CreateExistingBudgetAction")
        properties = action["Properties"]
        self.assertEqual(properties["ApprovalModel"], "AUTOMATIC")
        self.assertEqual(properties["NotificationType"], "ACTUAL")
        self.assertEqual(properties["ActionThreshold"], {"Type": "PERCENTAGE", "Value": 100})
        stop_policy = self.resources["BudgetStopPolicy"]["Properties"]["PolicyDocument"]
        encoded = json.dumps(stop_policy)
        self.assertIn("batch:SubmitJob", encoded)
        self.assertIn("cloudformation:*", encoded)

    def test_budget_email_is_required_and_not_personal_source_data(self) -> None:
        parameter = self.template["Parameters"]["BudgetAlertEmail"]
        self.assertNotIn("Default", parameter)
        self.assertNotIn("@gmail.com", self.raw)
        deploy = DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('BUDGET_EMAIL="${BUDGET_EMAIL:-}"', deploy)
        self.assertIn("Set BUDGET_EMAIL", deploy)
        workflow = (REPO_ROOT / ".github" / "workflows" / "aws-deploy.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("vars.AWS_BUDGET_ALERT_EMAIL", workflow)
        self.assertIn('"BudgetAlertEmail=${BUDGET_EMAIL}"', workflow)
        self.assertIn("github.actor_id == '125384326'", workflow)
        self.assertIn("github.event_name == 'push' || inputs.build_image", workflow)

    def test_container_versions_and_dino_ref_are_pinned(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        self.assertIn("pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime", dockerfile)
        self.assertIn("0279f7aa29974bf64e61d0ff6e979b41a249b3662a46e30778dbf80b8c99c361", dockerfile)
        self.assertIn("torchvision==0.18.0", dockerfile)
        self.assertIn("platform.python_version().startswith('3.10.')", dockerfile)
        self.assertIn("MUJOCO_GL=osmesa", dockerfile)
        self.assertIn("MUJOCO_PY_FORCE_CPU=1", dockerfile)
        self.assertIn(
            "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6", dockerfile
        )
        self.assertIn("EXPECTED_RUN_UID=10001", dockerfile)
        self.assertIn("7764ea0f912e53c92e82eb78a2a1631e92725fc8", dockerfile)
        self.assertIn("a436ca2f4144c38b837205635bbd60ffe1162d5b44c87df22232795978d7d012", dockerfile)
        self.assertIn("1fe665267a6149dfb8551cec52b419fa6e82533fab6dd7678939209246e792ee", dockerfile)
        self.assertIn("sha256sum --check --strict", dockerfile)
        self.assertIn("org.opencontainers.image.revision", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn('CMD ["python", "-m", "infra.container.smoke_test"]', dockerfile)
        self.assertIn("mujocopy-buildlock", dockerfile)
        self.assertIn("chown tsrunner:tsrunner", dockerfile)
        self.assertLess(
            dockerfile.index("import mujoco_py"),
            dockerfile.index("USER 10001:10001"),
            "the legacy extension must be built while site-packages is writable",
        )
        self.assertIn("mkdir -p /scratch", dockerfile)
        self.assertNotIn("MUJOCO_PY_MUJOCO_PATH=/root", dockerfile)
        dockerignore = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
        self.assertIn("!infra/experiments/**", dockerignore)
        self.assertTrue((REPO_ROOT / "infra" / "experiments" / "runner.py").is_file())
        self.assertTrue(
            (REPO_ROOT / "infra" / "experiments" / "manifests" / "umaze_exact_anchor.yaml").is_file()
        )

    def test_container_smoke_exercises_legacy_pointmaze_as_non_root(self) -> None:
        smoke = (REPO_ROOT / "infra" / "container" / "smoke_test.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("os.geteuid()", smoke)
        self.assertIn("PointMazeWrapper", smoke)
        self.assertIn("environment.prepare", smoke)
        self.assertIn("environment.step", smoke)
        self.assertIn("mujoco_py.cymj.__file__", smoke)
        self.assertIn("mujocopy-buildlock", smoke)
        self.assertIn("LD_PRELOAD", smoke)
        self.assertIn("torch.backends.mkldnn.enabled = False", smoke)
        self.assertIn('"mkldnn_enabled": torch.backends.mkldnn.enabled', smoke)
        self.assertIn("visual.shape != (224, 224, 3)", smoke)

    def test_deploy_preserves_oidc_provider_ownership(self) -> None:
        deploy = DEPLOY_SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("Preserve ownership across updates", deploy)
        self.assertIn("ParameterKey=='GitHubOidcProviderArn'", deploy)


if __name__ == "__main__":
    unittest.main()
