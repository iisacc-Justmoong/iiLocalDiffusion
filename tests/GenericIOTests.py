#!/usr/bin/env python3
"""Typed generic tensor interchange; real runtime round trips when available."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate_any as generic
import generic_io

try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None


class GenericIOContractTests(unittest.TestCase):
    def test_output_specs_require_explicit_semantics_layout_and_representation(self):
        spec = {"images": {"semantic": "latents", "layout": "BSC", "representation_space": "vae-v1"}}
        self.assertEqual(generic_io.validate_output_specs(spec), spec)
        for invalid in ({"images": {}}, {"images": {**spec["images"], "extra": 1}},
                        {"images": {**spec["images"], "layout": ""}},
                        {"images": {**spec["images"], "semantic": "unknown"}},
                        {"sequences": spec["images"]},
                        {"past_key_values": spec["images"]}, {"kv_cache": spec["images"]}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generic_io.validate_output_specs(invalid)

    def test_input_descriptor_rejects_missing_keys_pickle_and_nonfinite_contracts(self):
        descriptor = {"tensor_path": "/absolute/test.safetensors", "key": "x",
                      "semantic": "latents", "layout": "BSC", "representation_space": "vae-v1"}
        self.assertEqual(generic_io.validate_input_descriptor(descriptor), descriptor)
        for invalid in ({**descriptor, "tensor_path": "test.pt"}, {**descriptor, "key": ""},
                        {**descriptor, "sha256": "main"}, {**descriptor, "shape": [1, True]},
                        {**descriptor, "shape": [1, 0]}, {**descriptor, "unexpected": 1},
                        {k: v for k, v in descriptor.items() if k != "representation_space"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generic_io.validate_input_descriptor(invalid)


@unittest.skipIf(torch is None, "optional Torch/safetensors runtime is unavailable")
class GenericIORuntimeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="generic-io-tests-")
        self.directory = Path(self.temporary.name)
        self.spec = {"semantic": "latents", "layout": "BSC", "representation_space": "vae-v1"}

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, **changes):
        return SimpleNamespace(output_dir=self.directory / "outputs", overwrite=False,
                               tensor_outputs=changes.get("tensor_outputs", {}),
                               inputs=changes.get("inputs", {}), video_layout="auto", audio_sample_rate=None)

    def descriptor(self, tensor):
        path = self.directory / "input.safetensors"
        save_file({"exact": tensor, "other": torch.zeros(1)}, str(path))
        return {"tensor_path": str(path), "key": "exact", **self.spec}

    def test_latent_bfloat16_round_trip_preserves_dtype_shape_values_and_descriptor(self):
        original = torch.arange(24, dtype=torch.bfloat16).reshape(1, 6, 4)
        args = self.args(tensor_outputs={"images": self.spec}, inputs={"output_type": "latent"})
        outputs = generic.save_outputs({"images": original}, None, args, None, None)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["kind"], "latent")
        descriptor = outputs[0]["tensor_input"]
        self.assertEqual(descriptor["dtype"], "bfloat16")
        self.assertEqual(descriptor["shape"], [1, 6, 4])
        values, identities = generic.prepare_inputs({"image": descriptor}, None)
        self.assertTrue(torch.equal(values["image"], original))
        self.assertEqual(values["image"].dtype, torch.bfloat16)
        self.assertEqual(identities[0]["sha256"], descriptor["sha256"])

    def test_sequences_and_tokens_preserve_int64_and_are_not_iterated_as_media(self):
        sequences = torch.tensor([[2**53 + 7, 42]], dtype=torch.int64)
        tokens = torch.tensor([2, 3, 4], dtype=torch.int32)
        outputs = generic.save_outputs({"sequences": sequences, "token_ids": tokens, "text": "ok"},
                                       None, self.args(), None, None)
        self.assertEqual(len(outputs), 3)
        by_name = {item["field"]: item for item in outputs}
        for name, value in (("sequences", sequences), ("token_ids", tokens)):
            actual = load_file(by_name[name]["path"])[name]
            self.assertTrue(torch.equal(actual, value))
            self.assertEqual(actual.dtype, value.dtype)

    def test_input_requires_exact_key_hash_dtype_and_shape_when_declared(self):
        descriptor = self.descriptor(torch.ones(1, 2, 3))
        for invalid in ({**descriptor, "key": "missing"}, {**descriptor, "sha256": "a" * 64},
                        {**descriptor, "dtype": "int64"}, {**descriptor, "shape": [1, 3, 2]}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                generic.prepare_inputs({"latents": invalid}, None)

    def test_tensor_input_changed_during_load_fails(self):
        descriptor = self.descriptor(torch.ones(1, 2, 3))
        real_identity = generic.file_identity
        count = 0

        def changing_identity(path):
            nonlocal count
            count += 1
            result = real_identity(path)
            if count > 1:
                result["sha256"] = "0" * 64
            return result

        with patch.object(generic, "file_identity", changing_identity), self.assertRaisesRegex(RuntimeError, "changed"):
            generic.prepare_inputs({"latents": descriptor}, None)

    def test_malformed_tensor_outputs_do_not_publish_success_reports(self):
        for result, specs in (({"sequences": torch.tensor([[1.5]])}, {}),
                              ({"sequences": torch.tensor([[True]])}, {}),
                              ({"images": torch.tensor([float("nan")])}, {"images": self.spec}),
                              ({"images": torch.ones(1, 2)}, {"images": self.spec}),
                              ({"images": torch.ones(1, 2, 3)}, {"missing": self.spec})):
            with self.subTest(result=result), self.assertRaises(ValueError):
                generic.publish_generation(result, None, self.args(tensor_outputs=specs), None, None, {})
            self.assertFalse((self.directory / "outputs/generation.json").exists())

    def test_opaque_cache_is_never_serialized_while_sequences_remain_usable(self):
        outputs = generic.save_outputs({"sequences": torch.tensor([[1, 2]]), "past_key_values": object()},
                                       None, self.args(), None, None)
        self.assertEqual([item["field"] for item in outputs], ["sequences"])

    def test_missing_tensor_output_and_integer_semantic_checks_do_not_create_artifacts(self):
        with self.assertRaisesRegex(ValueError, "did not return"):
            generic.save_outputs({}, None, self.args(tensor_outputs={"latents": self.spec}), None, None)
        for value in (torch.tensor([[-1, 2]]), torch.tensor([[1.0, 2.0]]), torch.tensor([[True, False]])):
            with self.subTest(dtype=value.dtype), self.assertRaises(ValueError):
                generic.save_outputs({"token_ids": value}, None, self.args(), None, None)
        self.assertFalse((self.directory / "outputs").exists())

    def test_tensor_publication_collision_retains_previous_bytes(self):
        args = self.args(tensor_outputs={"images": self.spec})
        first = generic.save_outputs({"images": torch.ones(1, 2, 3)}, None, args, None, None)[0]
        before = Path(first["path"]).read_bytes()
        with self.assertRaises(FileExistsError):
            generic.save_outputs({"images": torch.zeros(1, 2, 3)}, None, args, None, None)
        self.assertEqual(Path(first["path"]).read_bytes(), before)
        self.assertEqual(list(args.output_dir.glob(".*.tmp")), [])

    def test_iild_tensor_metadata_cannot_be_relabelled_on_input(self):
        outputs = generic.save_outputs({"images": torch.ones(1, 2, 3)}, None,
                                       self.args(tensor_outputs={"images": self.spec}), None, None)
        descriptor = outputs[0]["tensor_input"]
        for changed in ({"semantic": "velocity"}, {"layout": "BCS"},
                        {"representation_space": "different-vae-space"}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "stored.*metadata"):
                generic.prepare_inputs({"latents": {**descriptor, **changed}}, None)

    def test_diffusion_epsilon_v_prediction_and_flow_velocity_are_distinct(self):
        specs = {name: {**self.spec, "semantic": semantic} for name, semantic in
                 (("noise_prediction", "epsilon"), ("diffusion_v", "v-prediction"), ("flow_v", "velocity"))}
        outputs = generic.save_outputs({name: torch.zeros(1, 2, 3) for name in specs}, None,
                                       self.args(tensor_outputs=specs), None, None)
        self.assertEqual([item["tensor_input"]["semantic"] for item in outputs],
                         ["epsilon", "v-prediction", "velocity"])


if __name__ == "__main__":
    unittest.main()
