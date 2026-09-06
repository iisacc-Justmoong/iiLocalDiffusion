#!/usr/bin/env python3
"""Behavioral contracts for ordered, typed generation composition."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference/diffusers"))
from generation_composition import Port, Source, Stage, GenerationPipeline


class CompositionTests(unittest.TestCase):
    def test_file_descriptors_map_semantics_and_require_known_space(self):
        base = {"tensor_path": "/unused.safetensors", "key": "value", "dtype": "float32",
                "shape": [1, 2], "layout": "BC", "representation_space": "vae-v1"}
        for semantic, expected in (("latents", "latent"), ("velocity", "velocity"),
                                   ("v-prediction", "v_prediction"), ("tensor", "tensor")):
            port = Port.from_tensor_descriptor({**base, "semantic": semantic})
            self.assertEqual(port.semantic, expected)
        for space in ("unspecified", "auto", "unknown"):
            with self.subTest(space=space), self.assertRaisesRegex(ValueError, "representation_space"):
                Port.from_tensor_descriptor({**base, "semantic": "latents", "representation_space": space})

    def text(self):
        return Port.text()

    def stage(self, name="generate", architecture="diffusion", input_port=None,
              output_port=None, source=None, execute=None):
        return Stage(name, architecture, {"input": input_port or self.text()},
                     {"output": output_port or self.text()},
                     {"input": source or Source.input("prompt")},
                     execute or (lambda values: {"output": values["input"] + "!"}))

    def test_all_architectures_and_mixed_order_execute(self):
        stages = []
        names = ("diffusion", "rectified-flow", "flow-matching", "autoregressive")
        for index, kind in enumerate(names):
            stages.append(self.stage(str(index), kind,
                source=Source.input("prompt") if index == 0 else Source(str(index - 1), "output")))
        pipeline = GenerationPipeline({"prompt": self.text()}, stages, {"text": Source("3", "output")})
        result = pipeline.run({"prompt": "hello"})
        self.assertEqual(result.outputs, {"text": "hello!!!!"})
        self.assertEqual(result.architectures, names)
        self.assertTrue(result.hybrid)

    def test_incompatible_space_fails_before_any_executor(self):
        first = Port.tensor("float32", (1, 4), "BC", "latent", "vae-a")
        second = Port.tensor("float32", (1, 4), "BC", "latent", "vae-b")
        execute = Mock()
        with self.assertRaisesRegex(ValueError, "representation_space"):
            GenerationPipeline({"prompt": self.text()}, [
                self.stage(output_port=first, execute=execute),
                self.stage("next", input_port=second, source=Source("generate", "output"))],
                {"text": Source("next", "output")})
        execute.assert_not_called()

    def test_semantic_dtype_layout_and_shape_are_not_silently_converted(self):
        port = Port.tensor("float32", (1, 4), "BC", "velocity", "state-a")
        variants = [("float32", (1, 4), "BC", "v_prediction", "state-a"),
                    ("float64", (1, 4), "BC", "velocity", "state-a"),
                    ("float32", (1, 4), "CB", "velocity", "state-a"),
                    ("float32", (1, 5), "BC", "velocity", "state-a")]
        for fields in variants:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                port.require_compatible(Port.tensor(*fields))

    def test_forward_reference_missing_duplicate_unused_and_bad_outputs_rejected(self):
        invalid = [
            [self.stage(source=Source("future", "output"))],
            [self.stage(source=Source.input("missing"))],
            [self.stage(), self.stage()],
        ]
        for stages in invalid:
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                GenerationPipeline({"prompt": self.text()}, stages, {"text": Source("generate", "output")})
        with self.assertRaises(ValueError):
            GenerationPipeline({"prompt": self.text()}, [self.stage()], {"text": Source("generate", "missing")})

    def test_actual_bad_output_stops_downstream(self):
        downstream = Mock()
        pipeline = GenerationPipeline({"prompt": self.text()}, [
            self.stage(execute=lambda _: {"output": 3}),
            self.stage("next", source=Source("generate", "output"), execute=downstream)],
            {"text": Source("next", "output")})
        with self.assertRaisesRegex(ValueError, "generate.output"):
            pipeline.run({"prompt": "test"})
        downstream.assert_not_called()

    def test_unknown_missing_inputs_and_outputs_fail(self):
        for supplied in ({}, {"prompt": "hi", "extra": 1}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                GenerationPipeline({"prompt": self.text()}, [self.stage()],
                    {"text": Source("generate", "output")}).run(supplied)
        for output in ({}, {"output": "ok", "unrequested": "bad"}):
            with self.subTest(output=output), self.assertRaises(ValueError):
                GenerationPipeline({"prompt": self.text()}, [self.stage(execute=lambda _: output)],
                    {"text": Source("generate", "output")}).run({"prompt": "hi"})

    def test_invalid_port_and_architecture_fail_early(self):
        for args in (("float32", (0,), "B", "latent", "vae"),
                     ("float32", (True,), "B", "latent", "vae"),
                     ("float32", (2,), "BC", "latent", "vae"),
                     ("float32", (2,), "B", "token_ids", "tok"),
                     ("int64", (1, 2), "BC", "sample", "state"),
                     ("float32", (2,), "B", "latent", "")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Port.tensor(*args)
        with self.assertRaises(ValueError):
            GenerationPipeline({"prompt": self.text()}, [self.stage(architecture="anything")],
                {"text": Source("generate", "output")})


if __name__ == "__main__":
    unittest.main()
