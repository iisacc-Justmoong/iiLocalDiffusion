#!/usr/bin/env python3
"""Bounded real-container fixtures for downloaded model identity and routing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import downloaded_model


class DownloadedModelTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="downloaded-model-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def safetensors(self, shapes, metadata=None, name="unrelated-name.safetensors"):
        header, cursor = {}, 0
        if metadata:
            header["__metadata__"] = metadata
        for key, shape in shapes.items():
            end = cursor + math.prod(shape) * 4
            header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [cursor, end]}
            cursor = end
        raw = json.dumps(header).encode()
        path = self.directory / name
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))
        return path

    def sd(self, context=768, channels=4, **kwargs):
        return self.safetensors({
            "model.diffusion_model.input_blocks.0.0.weight": [1, channels, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, context],
        }, **kwargs)

    def flux(self, guidance=True, **kwargs):
        shapes = {"double_blocks.0.img_attn.norm.key_norm.scale": [1], "img_in.weight": [1, 64]}
        if guidance:
            shapes["guidance_in.in_layer.weight"] = [1, 1]
        return self.safetensors(shapes, **kwargs)

    def info(self, model, base="SD 1.5", role="Checkpoint", sha=True, **extra):
        info = {"baseModel": base, "model": {"type": role}}
        if sha:
            info["files"] = [{"name": "original-download-name.safetensors", "hashes": {
                "SHA256": hashlib.sha256(model.read_bytes()).hexdigest().upper()}}]
        info.update(extra)
        path = model.with_suffix(".civitai.info")
        path.write_text(json.dumps(info))
        return path

    def test_tensor_sd_families_route_without_name_guessing(self):
        for context, architecture, preset, pipeline in (
                (768, "sd1", "sd15-compatible", "StableDiffusionPipeline"),
                (1024, "sd2", None, "StableDiffusionPipeline"),
                (2048, "sdxl", "sdxl", "StableDiffusionXLPipeline")):
            with self.subTest(context=context):
                found = downloaded_model.inspect_downloaded_model(self.sd(context))
                self.assertEqual((found["architecture"], found["preset"], found["pipeline_class"]), (architecture, preset, pipeline))
                self.assertEqual(found["weights_role"], "denoiser")
                self.assertEqual(found["missing_components"], ["vae", "text_encoder"])
                self.assertIsNone(found["base_model"])

    def test_full_checkpoint_components_are_distinguished_from_denoiser(self):
        path = self.safetensors({
            "model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
            "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 768],
            "first_stage_model.encoder.conv_in.weight": [1],
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight": [1],
        })
        found = downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(found["weights_role"], "checkpoint")
        self.assertEqual(found["missing_components"], [])

    def test_illustrious_metadata_keeps_architecture_and_verifies_renamed_file_hash(self):
        path = self.sd(2048)
        self.info(path, "Illustrious")
        found = downloaded_model.inspect_downloaded_model(path)
        self.assertEqual((found["base_model"], found["preset"], found["confidence"]), ("Illustrious", "illustrious", "exact"))

    def test_noobai_prediction_metadata_routes_v_pred(self):
        path = self.sd(2048, metadata={"ss_v_parameterization": "true"})
        self.info(path, "NoobAI")
        found = downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(found["prediction_type"], "v_prediction")
        self.assertEqual(found["preset"], "noobai-v-pred")

    def test_flux_guidance_distinguishes_dev_and_schnell(self):
        for guidance, preset in ((True, "flux1-dev"), (False, "flux1-schnell-compatible")):
            with self.subTest(guidance=guidance):
                self.assertEqual(downloaded_model.inspect_downloaded_model(self.flux(guidance))["preset"], preset)

    def test_flux_dev_metadata_rejects_schnell_weights(self):
        path = self.flux(False)
        self.info(path, "Flux.1 D")
        with self.assertRaisesRegex(ValueError, "guidance embeddings"):
            downloaded_model.inspect_downloaded_model(path)

    def test_krea_reuses_dev_architecture_only_when_metadata_matches(self):
        path = self.flux()
        self.info(path, "Flux.1 Krea")
        self.assertEqual(downloaded_model.inspect_downloaded_model(path)["preset"], "flux1-krea-dev")

    def test_sd3_joint_attention_routes_to_sd3_pipeline(self):
        path = self.safetensors({"model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias": [1]})
        self.assertEqual(downloaded_model.inspect_downloaded_model(path)["pipeline_class"], "StableDiffusion3Pipeline")

    def test_inpainting_input_channels_preserve_required_image_task(self):
        found = downloaded_model.inspect_downloaded_model(self.sd(2048, channels=9))
        self.assertEqual(found["task"], "inpainting")
        self.assertEqual(found["pipeline_class"], "StableDiffusionXLInpaintPipeline")
        self.assertIsNone(found["preset"])

    def test_refiner_never_routes_to_text_to_image_base(self):
        found = downloaded_model.inspect_downloaded_model(self.sd(1280))
        self.assertEqual(found["pipeline_class"], "StableDiffusionXLImg2ImgPipeline")
        self.assertEqual(found["task"], "image-to-image")

    def test_adapter_roles_never_select_checkpoint_pipeline(self):
        cases = (
            ({"lora_unet_input_blocks_1_attn1.to_q.lora_down.weight": [2, 2]}, "lora"),
            ({"transformer.q_proj.lora_A.weight": [2, 2]}, "lora"),
            ({"encoder.conv_in.weight": [1], "decoder.conv_out.weight": [1]}, "vae"),
            ({"emb_params": [1, 768]}, "embedding"),
            ({"control_model.time_embed.0.weight": [1]}, "controlnet"),
        )
        for shapes, role in cases:
            with self.subTest(role=role):
                found = downloaded_model.inspect_downloaded_model(self.safetensors(shapes))
                self.assertEqual(found["role"], role)
                self.assertIsNone(found["preset"])
                self.assertIsNone(found["pipeline_class"])
                self.assertIn("--", found["role_guidance"])

    def test_false_checkpoint_sidecar_cannot_override_lora_tensors(self):
        path = self.safetensors({"lora_unet_to_q.lora_down.weight": [2, 2]})
        self.info(path)
        with self.assertRaisesRegex(ValueError, "role.*conflicts"):
            downloaded_model.inspect_downloaded_model(path)

    def test_wrong_sha256_rejects_matching_model_name(self):
        path = self.sd()
        self.info(path, files=[{"hashes": {"SHA256": "0" * 64}}])
        with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
            downloaded_model.inspect_downloaded_model(path)

    def test_wrong_architecture_metadata_is_rejected_even_with_matching_hash(self):
        path = self.sd()
        self.info(path, "Illustrious")
        with self.assertRaisesRegex(ValueError, "conflicts with tensor architecture"):
            downloaded_model.inspect_downloaded_model(path)

    def test_unknown_category_rejects_instead_of_guessing(self):
        path = self.sd()
        self.info(path, "SDXL Mega Secret 999")
        with self.assertRaisesRegex(ValueError, "Unknown Civitai"):
            downloaded_model.inspect_downloaded_model(path)

    def test_unknown_tensors_are_never_guessed_from_filename(self):
        found = downloaded_model.inspect_downloaded_model(self.safetensors({"unrecognized": [2]}, name="Illustrious-SDXL-flux.safetensors"))
        self.assertEqual(found["role"], "unknown")
        self.assertIsNone(found["architecture"])
        self.assertIsNone(found["preset"])

    def test_explicit_info_path_and_unverified_metadata_confidence(self):
        path = self.sd()
        info = self.info(path, sha=False)
        renamed = self.directory / "saved-version.json"
        info.rename(renamed)
        found = downloaded_model.inspect_downloaded_model(path, renamed)
        self.assertEqual(found["confidence"], "metadata")
        self.assertEqual(found["model_info"], str(renamed))

    def test_ambiguous_automatic_sidecars_require_explicit_path(self):
        path = self.sd()
        self.info(path)
        Path(str(path) + ".civitai.info").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Multiple Civitai"):
            downloaded_model.inspect_downloaded_model(path)

    def test_truncation_and_oversized_header_are_rejected(self):
        for content in (b"short", struct.pack("<Q", downloaded_model.MAX_HEADER_BYTES + 1), struct.pack("<Q", 100) + b"{}"):
            path = self.directory / "bad.safetensors"
            path.write_bytes(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                downloaded_model.inspect_downloaded_model(path)

    def test_duplicate_json_keys_are_rejected(self):
        path = self.directory / "duplicate.safetensors"
        raw = b'{"x":{},"x":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            downloaded_model.inspect_downloaded_model(path)

    def test_tensor_extents_are_validated_without_decoding(self):
        path = self.safetensors({"weight": [1]})
        path.write_bytes(path.read_bytes() + b"trailing")
        with self.assertRaisesRegex(ValueError, "trailing unindexed"):
            downloaded_model.inspect_downloaded_model(path)

    def test_malformed_tensor_descriptor_fails_clearly(self):
        for descriptor in ({"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]},
                           {"dtype": [], "shape": [1], "data_offsets": [0, 4]},
                           {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}):
            raw = json.dumps({"weight": descriptor}).encode()
            path = self.directory / "malformed.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"1234")
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                downloaded_model.inspect_downloaded_model(path)

    def gguf(self, architecture="flux", name="gguf.bin", metadata_type=8):
        def string(value):
            data = value.encode()
            return struct.pack("<Q", len(data)) + data
        raw = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
        raw += string("general.architecture") + struct.pack("<I", metadata_type) + string(architecture)
        raw += string("unknown_tensor") + struct.pack("<IQIQ", 1, 1, 0, 0)
        raw += bytes((-len(raw)) % 32) + bytes(32)
        path = self.directory / name
        path.write_bytes(raw)
        return path

    def test_gguf_reads_architecture_without_loading_payload(self):
        found = downloaded_model.inspect_downloaded_model(self.gguf(name="weights.gguf"))
        self.assertEqual(found["architecture"], "flux1")
        self.assertEqual(found["format"], "gguf")
        self.assertEqual(found["weights_role"], "denoiser")
        self.assertEqual(found["pipeline_class"], "FluxPipeline")

    def test_gguf_truncated_payload_rejects_outside_offsets(self):
        path = self.gguf(name="truncated.gguf")
        path.write_bytes(path.read_bytes()[:-32])
        with self.assertRaisesRegex(ValueError, "offset"):
            downloaded_model.inspect_downloaded_model(path)

    def test_gguf_unknown_metadata_type_is_rejected(self):
        path = self.gguf(name="unknown-kind.gguf", metadata_type=99)
        with self.assertRaisesRegex(ValueError, "metadata value type"):
            downloaded_model.inspect_downloaded_model(path)

    def test_weights_only_loading_is_explicit_and_never_falls_back(self):
        path = self.directory / "local.ckpt"
        path.write_bytes(b"not read by mocked torch")
        calls = []
        def reject(*args, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("unsafe GLOBAL")
        with patch.dict(sys.modules, {"torch": SimpleNamespace(load=reject, __version__="2.10.0")}):
            with self.assertRaisesRegex(ValueError, "unsafe pickle loading is not permitted"):
                downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(calls, [{"map_location": "meta", "weights_only": True}])

    def test_vulnerable_torch_is_rejected_before_loading(self):
        path = self.directory / "old-runtime.ckpt"
        path.write_bytes(b"must never deserialize")
        for version in ("2.6.0", "2.9.1", "2.10.0rc1", "unknown"):
            with self.subTest(version=version), patch.dict(sys.modules, {"torch": SimpleNamespace(__version__=version)}):
                with self.assertRaises(ValueError):
                    downloaded_model.inspect_downloaded_model(path)

    def test_missing_torch_does_not_execute_pickle(self):
        path = self.directory / "local.ckpt"
        path.write_bytes(b"pickle must never execute")
        self.info(path)
        with patch.dict(sys.modules, {"torch": None}):
            found = downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(found["preset"], "sd15-compatible")
        self.assertTrue(any("was not opened" in evidence for evidence in found["evidence"]))

    def test_hosted_category_is_rejected(self):
        from civitai_catalog import list_base_models
        category = next(record["name"] for record in list_base_models() if record["local_status"] == "hosted")
        path = self.sd()
        self.info(path, category)
        with self.assertRaisesRegex(ValueError, "cloud-only"):
            downloaded_model.inspect_downloaded_model(path)


if __name__ == "__main__":
    unittest.main()
