#!/usr/bin/env python3
"""Check manual inference fixtures without executing a model or a GUI."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "BackendRuntimeSmoke.py", "FamilyVaeDiffusersSmoke.py", "QwenVaeDiffusersSmoke.py",
    "HalfResolutionSmoke.py", "DeforumDiffusersSmoke.py", "DefaultModifierDiffusersSmoke.py",
    "InterpolatorDiffusersSmoke.py", "InterpolatorAdapterSmoke.py",
    "UniversalLoraDiffusersSmoke.py", "VideoDiffusersSmoke.py",
)


class SmokeLayoutTests(unittest.TestCase):
    def test_manual_inference_outputs_do_not_use_the_build_tree(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                tree = ast.parse((ROOT / "tests" / name).read_text(encoding="utf-8"))
                bad = []
                resource_outputs = []
                for node in ast.walk(tree):
                    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                            and isinstance(node.left, ast.Name) and node.left.id == "ROOT"
                            and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
                        path = Path(node.right.value)
                        if path.parts and path.parts[0] == "build":
                            bad.append((node.lineno, node.right.value))
                        if path.parts[:3] == ("resources", "outputs", "verification"):
                            resource_outputs.append(path)
                self.assertEqual(bad, [], f"{name} writes fixtures or reports under build/")
                self.assertTrue(resource_outputs, f"{name} must choose a project resource output")


if __name__ == "__main__":
    unittest.main()
