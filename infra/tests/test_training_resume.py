"""Static contracts for interruption-safe training resume behavior."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


TRAIN_PATH = Path(__file__).resolve().parents[2] / "train.py"


def _self_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == name
    )


class TrainingResumeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TRAIN_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        trainer = next(
            node
            for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Trainer"
        )
        cls.methods = {
            node.name: node
            for node in trainer.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_restored_action_and_proprio_modules_are_not_reinstantiated(self) -> None:
        init_models = self.methods["init_models"]
        for attribute in ("action_encoder", "proprio_encoder"):
            guarded_assignments = []
            for node in ast.walk(init_models):
                if not isinstance(node, ast.If):
                    continue
                test = node.test
                if not (
                    isinstance(test, ast.Compare)
                    and _self_attribute(test.left, attribute)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Is)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                ):
                    continue
                guarded_assignments.extend(
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.Assign)
                    and any(_self_attribute(target, attribute) for target in child.targets)
                )
            self.assertTrue(
                guarded_assignments,
                f"{attribute} must only be instantiated when no restored module exists",
            )

    def test_action_proprio_optimizer_is_saved_and_restored(self) -> None:
        init_source = ast.get_source_segment(
            self.source, self.methods["__init__"]
        ) or ""
        optimizer_source = ast.get_source_segment(
            self.source, self.methods["init_optimizers"]
        ) or ""
        self.assertIn('"action_encoder_optimizer"', init_source)
        self.assertIn('"action_encoder_optimizer"', optimizer_source)
        self.assertIn("load_state_dict", optimizer_source)


if __name__ == "__main__":
    unittest.main()
