#!/usr/bin/env python3
"""Bounded real-container fixtures for downloaded model identity and routing."""

from __future__ import annotations
import resource_fixture

import hashlib
import json
import math
import pickle
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import downloaded_model
from _file_provider import provider


class DownloadedModelTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = resource_fixture.TemporaryDirectory(prefix="downloaded-model-")
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

    def test_sharded_safetensors_index_identifies_model_from_all_shards(self):
        first = self.safetensors({'model.diffusion_model.input_blocks.0.0.weight': [1, 4, 1, 1]},
                                 name='part-1.safetensors')
        second = self.safetensors({'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': [1, 2048]},
                                  name='part-2.safetensors')
        index = self.directory / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {
            'model.diffusion_model.input_blocks.0.0.weight': first.name,
            'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': second.name}}))
        found = downloaded_model.inspect_downloaded_model(index)
        self.assertEqual(found['format'], 'safetensors-index')
        self.assertEqual(found['architecture'], 'sdxl')
        self.assertEqual(found['role'], 'checkpoint')
        self.assertEqual(set(found['sharded_index']['shards']), {str(first), str(second)})
        self.assertIn('vae', found['backend_plan']['missing_components'])

    def test_sharded_index_cache_and_source_identity_track_shards(self):
        from inference_session import InferenceSession
        first = self.safetensors({'model.diffusion_model.input_blocks.0.0.weight': [1, 4, 1, 1]},
                                 name='part-1.safetensors')
        second = self.safetensors({'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': [1, 2048]},
                                  name='part-2.safetensors')
        index = self.directory / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {
            'model.diffusion_model.input_blocks.0.0.weight': first.name,
            'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': second.name}}))
        with InferenceSession():
            self.assertEqual(downloaded_model.inspect_downloaded_model(index)['architecture'], 'sdxl')
            self.safetensors({'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': [1, 768]},
                             name=second.name)
            self.assertEqual(downloaded_model.inspect_downloaded_model(index)['architecture'], 'sd1')

    def test_sharded_index_suffix_is_case_insensitive(self):
        first = self.safetensors({'model.diffusion_model.input_blocks.0.0.weight': [1, 4, 1, 1]},
                                 name='part-1.safetensors')
        second = self.safetensors({'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': [1, 2048]},
                                  name='part-2.safetensors')
        index = self.directory / 'model.SAFETENSORS.Index.JSON'
        index.write_text(json.dumps({'weight_map': {
            'model.diffusion_model.input_blocks.0.0.weight': first.name,
            'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': second.name}}))
        found = downloaded_model.inspect_downloaded_model(index)
        self.assertEqual((found['format'], found['architecture']), ('safetensors-index', 'sdxl'))

    def test_renamed_index_is_detected_from_its_safetensors_shards(self):
        first = self.safetensors({'model.diffusion_model.input_blocks.0.0.weight': [1, 4, 1, 1]},
                                 name='part-1.safetensors')
        second = self.safetensors({'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': [1, 2048]},
                                  name='part-2.safetensors')
        mapping = {'weight_map': {
            'model.diffusion_model.input_blocks.0.0.weight': first.name,
            'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight': second.name}}
        for name in ('renamed.index.json', 'renamed.bin', 'renamed.model'):
            with self.subTest(name=name):
                index = self.directory / name
                index.write_text(json.dumps(mapping))
                report = downloaded_model.inspect_downloaded_model(index)
                self.assertEqual((report['format'], report['architecture']), ('safetensors-index', 'sdxl'))
                self.assertEqual(set(report['sharded_index']['shards']), {str(first), str(second)})
        unsafe = self.directory / 'not-a-safetensors-index.bin'
        legacy = self.directory / 'legacy-shard.bin'
        legacy.write_bytes(b'PK\x03\x04' + bytes(32))
        unsafe.write_text(json.dumps({'weight_map': {'tensor': legacy.name}}))
        with self.assertRaisesRegex(ValueError, 'JSON model index'):
            downloaded_model.model_container_format(unsafe)

    def test_container_content_takes_precedence_over_filename(self):
        for name in ("checkpoint.bin", "checkpoint.custom", "checkpoint.gguf"):
            with self.subTest(name=name), patch.object(downloaded_model, "_torch_checkpoint",
                                                       side_effect=AssertionError("unsafe legacy reader selected")):
                found = downloaded_model.inspect_downloaded_model(self.sd(name=name))
                self.assertEqual((found["format"], found["architecture"]), ("safetensors", "sd1"))
        gguf = self.gguf(name="weights.bin")
        with patch.object(downloaded_model, "_torch_checkpoint", side_effect=AssertionError("legacy reader selected")):
            self.assertEqual(downloaded_model.inspect_downloaded_model(gguf)["format"], "gguf")

    def test_legacy_container_signature_takes_precedence_over_safe_suffix(self):
        for name, prefix in (("model.safetensors", b"PK\x03\x04"),
                             ("model.gguf", b"\x80\x02")):
            with self.subTest(name=name):
                path = self.directory / name
                path.write_bytes(prefix + bytes(16))
                self.assertEqual(downloaded_model.model_container_format(path), "pytorch")
                self.assertFalse(downloaded_model.is_supported_model_container(path))
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        header = header + b' ' * (640 - len(header))
        safe = self.directory / 'length-overlaps-pickle.bin'
        safe.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        self.assertEqual(safe.read_bytes()[:2], b'\x80\x02')
        self.assertEqual(downloaded_model.model_container_format(safe), 'safetensors')
        self.assertTrue(downloaded_model.is_supported_model_container(safe))

    def test_native_probe_recognizes_torch_archive_when_pickle_is_not_first(self):
        from image_package import probe_model_container
        archive = self.directory / 'renamed.data'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr('checkpoint/version', '3')
            output.writestr('checkpoint/data.pkl', b'\x80\x04fixture')
        self.assertEqual(downloaded_model.model_container_format(archive), 'pytorch')
        self.assertEqual(probe_model_container(archive), 'pytorch')
        raw = archive.read_bytes()
        eocd = raw.rfind(b'PK\x05\x06')
        self.assertGreater(eocd, 0)
        end = bytearray(raw[eocd:])
        count, size, offset = struct.unpack_from('<HII', end, 10)
        extended = struct.pack('<IQHHIIQQQQ', 0x06064b50, 44, 45, 45, 0, 0,
                               count, count, size, offset)
        locator = struct.pack('<IIQI', 0x07064b50, 0, eocd, 1)
        struct.pack_into('<HHII', end, 8, 0xffff, 0xffff, 0xffffffff, 0xffffffff)
        zip64 = self.directory / 'zip64.data'
        zip64.write_bytes(raw[:eocd] + extended + locator + end)
        self.assertEqual(probe_model_container(zip64), 'pytorch')
        broken = self.directory / 'broken-zip64.data'
        broken_data = bytearray(zip64.read_bytes())
        struct.pack_into('<Q', broken_data, eocd + len(extended) + 8, len(broken_data) + 1)
        broken.write_bytes(broken_data)
        self.assertEqual(probe_model_container(broken), 'unknown')
        unrelated = self.directory / 'unrelated.zip'
        with zipfile.ZipFile(unrelated, 'w') as output:
            output.writestr('archive/readme.txt', 'not a checkpoint')
        self.assertEqual(probe_model_container(unrelated), 'unknown')

    def test_native_probe_recognizes_legacy_torch_magic_across_pickle_protocols(self):
        from image_package import probe_model_container
        for protocol in range(6):
            with self.subTest(protocol=protocol):
                path = self.directory / f'legacy-{protocol}.data'
                path.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c, protocol=protocol) + b'fixture')
                self.assertEqual(probe_model_container(path), 'pytorch')
                self.assertEqual(downloaded_model.model_container_format(path), 'pytorch')
        generic = self.directory / 'other-pickle.data'
        generic.write_bytes(pickle.dumps({'not': 'weights'}, protocol=4))
        self.assertEqual(probe_model_container(generic), 'unknown')
        truncated = self.directory / 'truncated-legacy.data'
        truncated.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c, protocol=4)[:-1])
        self.assertEqual(probe_model_container(truncated), 'unknown')

    def test_pytorch_index_combines_exact_shard_shapes_for_routing(self):
        first = self.directory / 'first.data'
        second = self.directory / 'second.bin'
        content = pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture'
        first.write_bytes(content); second.write_bytes(content)
        first_name = 'model.diffusion_model.input_blocks.0.0.weight'
        second_name = 'model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight'
        index = self.directory / 'renamed.model'
        index.write_text(json.dumps({'weight_map': {first_name: first.name, second_name: second.name}}))
        shapes = {first: {first_name: (1, 4, 1, 1)}, second: {second_name: (1, 768)}}
        def inspect(path):
            return shapes[path], {}, ['weights-only fixture']
        with patch.object(downloaded_model, '_torch_checkpoint', side_effect=inspect):
            report = downloaded_model.inspect_downloaded_model(index)
            self.assertEqual((report['format'], report['architecture'], report['role']),
                             ('pytorch-index', 'sd1', 'checkpoint'))
            self.assertEqual(set(report['sharded_index']['shards']), {str(first), str(second)})
            shapes[second]['unexpected.weight'] = (1,)
            with self.assertRaisesRegex(ValueError, 'exactly'):
                downloaded_model.inspect_downloaded_model(index)

    def test_fp8_fnuz_and_complex_descriptors_are_inspected_without_conversion(self):
        for dtype, size in (('F8_E4M3FNUZ', 1), ('F8_E5M2FNUZ', 1), ('F8_E4M3FN', 1), ('C64', 8)):
            with self.subTest(dtype=dtype):
                header = {'tensor': {'dtype': dtype, 'shape': [1], 'data_offsets': [0, size]}}
                encoded = json.dumps(header).encode()
                path = self.directory / (dtype + '.safetensors')
                path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + bytes(size))
                shapes, _, _ = downloaded_model._safetensors(path)
                self.assertEqual(shapes['tensor'], (1,))

    def test_moss_transformer_identity_is_tensor_based_and_not_a_native_plan(self):
        for prefix in ("", "model.diffusion_model.", "diffusion_model.", "transformer."):
            with self.subTest(prefix=prefix):
                model = self.safetensors({
                    prefix + "video_embedder.weight": [4, 2],
                    prefix + "final_linear_video.weight": [2, 4],
                    prefix + "layers.0.proj.qkv.weight": [12, 4],
                }, name="renamed.safetensors")
                found = downloaded_model.inspect_downloaded_model(model)
                self.assertEqual((found["architecture"], found["role"], found["pipeline_class"]),
                                 ("moss-image", "checkpoint", "MOSSImagePipeline"))
                self.assertEqual(found["weights_role"], "denoiser")
                self.assertEqual(found["missing_components"], ["vae", "text_encoder"])
                self.assertNotIn("backend_plan", found)
        near_miss = self.safetensors({"video_embedder.weight": [4, 2],
            "final_linear_video.weight": [3, 4], "layers.0.proj.qkv.weight": [12, 4]})
        self.assertNotEqual(downloaded_model.inspect_downloaded_model(near_miss)["architecture"], "moss-image")
        mixed = self.safetensors({"video_embedder.weight": [4, 2],
            "model.diffusion_model.final_linear_video.weight": [2, 4],
            "transformer.layers.0.proj.qkv.weight": [12, 4]})
        self.assertNotEqual(downloaded_model.inspect_downloaded_model(mixed)["architecture"], "moss-image")

    def test_complete_anima_is_not_its_embedded_vae(self):
        for prefix in ("", "net.", "model.diffusion_model.net.", "diffusion_model."):
            with self.subTest(prefix=prefix):
                model = self.safetensors({
                    prefix + "llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2],
                    prefix + "blocks.0.self_attn.q_proj.weight": [2, 2],
                    "vae.encoder.conv1.weight": [1], "vae.decoder.conv1.weight": [1],
                    "text_encoders.llm.model.embed_tokens.weight": [2, 2],
                })
                found = downloaded_model.inspect_downloaded_model(model)
                self.assertEqual(found["role"], "checkpoint")
                self.assertEqual(found["architecture"], "anima")
                self.assertEqual(found["weights_role"], "checkpoint")
                self.assertEqual(found["missing_components"], [])

    def test_sdxl_prediction_markers_work_without_metadata_or_filename_hints(self):
        shapes = {"model.diffusion_model.input_blocks.0.0.weight": [1, 4, 1, 1],
                  "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 2048],
                  "v_pred": [0], "ztsnr": [0]}
        found = downloaded_model.inspect_downloaded_model(self.safetensors(shapes))
        self.assertEqual(found["prediction_type"], "v_prediction")
        self.assertTrue(found["zero_terminal_snr"])
        self.assertEqual(found["preset"], "sdxl")
        with self.assertRaisesRegex(ValueError, "Conflicting prediction"):
            downloaded_model.inspect_downloaded_model(self.safetensors(
                shapes, {"prediction_type": "epsilon"}, name="conflict.safetensors"))

    def test_anima_denoiser_and_vae_remain_distinct(self):
        found = downloaded_model.inspect_downloaded_model(self.safetensors({
            "llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2],
            "blocks.0.self_attn.q_proj.weight": [2, 2],
        }))
        self.assertEqual(found["architecture"], "anima")
        self.assertEqual(found["weights_role"], "denoiser")
        self.assertEqual(found["missing_components"], ["vae", "text_encoder"])
        found = downloaded_model.inspect_downloaded_model(self.safetensors({
            "vae.encoder.conv1.weight": [1], "vae.decoder.conv1.weight": [1],
        }, name="anima-complete.safetensors"))
        self.assertEqual(found["role"], "vae")
        self.assertIsNone(found["architecture"])

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

    def test_legacy_inspection_uses_an_open_stream_for_a_misleading_suffix(self):
        path = self.directory / 'legacy.safetensors'
        path.write_bytes(b'PK\x03\x04' + bytes(16))
        class Tensor:
            shape = (1, 4, 1, 1)
        calls = []
        def load(stream, **kwargs):
            calls.append((stream, kwargs, stream.read(4)))
            return {'model.diffusion_model.input_blocks.0.0.weight': Tensor()}
        with patch.dict(sys.modules, {'torch': SimpleNamespace(Tensor=Tensor, load=load, __version__='2.13.0')}):
            downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(calls[0][1], {'map_location': 'meta', 'weights_only': True})
        self.assertEqual(calls[0][2], b'PK\x03\x04')
        self.assertTrue(calls[0][0].closed)

    def test_legacy_inspection_detects_source_mutation(self):
        path = self.directory / 'legacy.ckpt'
        path.write_bytes(b'PK\x03\x04' + bytes(16))
        class Tensor:
            shape = (1, 4, 1, 1)
        def load(stream, **kwargs):
            path.write_bytes(b'PK\x03\x04' + b'x' * 16)
            return {'model.diffusion_model.input_blocks.0.0.weight': Tensor()}
        with patch.dict(sys.modules, {'torch': SimpleNamespace(Tensor=Tensor, load=load, __version__='2.13.0')}):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                downloaded_model.inspect_downloaded_model(path)

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
