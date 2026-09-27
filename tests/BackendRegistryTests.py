"""Real header fixtures exercise architecture detection, planning and publication."""
import ctypes as C
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import DownloadedModelTests as fixtures
import backend_registry as registry
import downloaded_model
import standalone_image
import native_image


SIGNATURES = {
    "sd2": {"input_blocks.0.0.weight": [1, 4, 1, 1], "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 1024]},
    "sd3": {"joint_blocks.0.context_block.adaLN_modulation.1.bias": [1]},
    "flux1-dev": {"double_blocks.0.img_attn.norm.key_norm.scale": [1], "img_in.weight": [1, 64], "guidance_in.in_layer.weight": [1, 1]},
    "flux1-schnell": {"double_blocks.0.img_attn.norm.key_norm.scale": [1], "img_in.weight": [1, 64]},
    "flux2": {"double_stream_modulation_img.lin.weight": [1, 1], "single_blocks.47.linear1.weight": [1, 1], "img_in.weight": [1, 128]},
    "flux2-klein": {"double_stream_modulation_img.linear.weight": [1, 1], "single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight": [1, 1]},
    "z-image": {"cap_embedder.0.weight": [1], "layers.0.attention.to_q.weight": [1, 1]},
    "qwen-image": {"transformer_blocks.0.img_mod.1.weight": [1, 1], "img_in.weight": [1, 64]},
    "chroma": {"distilled_guidance_layer.in_proj.weight": [1, 1], "img_in.weight": [1, 64]},
    "krea2": {"txtfusion.projector.weight": [1, 1], "img_in.weight": [1, 64]},
}


class BackendRegistryTests(unittest.TestCase):
    setUp = fixtures.DownloadedModelTests.setUp
    safetensors = fixtures.DownloadedModelTests.safetensors
    gguf = fixtures.DownloadedModelTests.gguf

    def component(self, name):
        return str(self.safetensors({"fixture.weight": [1]}, name=name + ".safetensors"))

    def request(self, architecture, **values):
        model = self.safetensors(SIGNATURES[architecture], name=architecture + ".safetensors")
        components = {slot: self.component(slot) for slot in ("vae", *registry.PROFILES[architecture][0])}
        return standalone_image.build_parser().parse_values({"model": str(model), "components": components,
            "prompt": "a tree", "width": 64, "height": 64, "default_modifiers": False,
            **({"krea2_variant": "raw"} if architecture == "krea2" else {}),
            **values})

    def test_krea2_variants_sampling_and_replay(self):
        for variant, steps, cfg in [('raw', 52, 4.5), ('turbo', 8, 1.0)]:
            _, args = standalone_image.resolve_arguments(self.request('krea2', krea2_variant=variant))
            self.assertEqual((args.steps, args.guidance_scale, args.native_sampler), (steps, cfg, 'euler'))
            self.assertTrue(args.native_v3)
            self.assertTrue(args.native_v2)
        _, args = standalone_image.resolve_arguments(self.request('krea2', krea2_variant='turbo',
            sigmas=[1, .5, .1], krea2_mu=0.0, native_sampler='heun'))
        self.assertEqual(args.native_sigmas, [1, .5, .1, 0])
        self.assertEqual(args.steps, 3)
        _, replay = standalone_image.resolve_arguments(standalone_image.build_parser().parse_values(
            native_image.configuration_values(args)))
        self.assertEqual(replay.native_sigmas, args.native_sigmas)
        self.assertEqual(replay.native_sampler, 'heun')
        _, automatic = standalone_image.resolve_arguments(self.request('krea2', krea2_variant='auto'))
        self.assertEqual(automatic.krea2_variant, 'raw')
        self.assertEqual(automatic.krea2['variant_source'], 'native-quality-default')
        for overrides in ({'width': 65}, {'width': 4096},
                          {'sigmas': [1, .5], 'steps': 3}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                standalone_image.resolve_arguments(self.request('krea2', **overrides))

    def test_krea2_quickgenerate_output_sizes_and_canvas_schedule(self):
        from krea2_contract import default_mu
        for width, height in ((1024, 1024), (1368, 1024), (1024, 1368),
                              (1824, 1024), (1024, 1824), (80, 64)):
            with self.subTest(size=(width, height)):
                _, args = standalone_image.resolve_arguments(self.request(
                    'krea2', width=width, height=height, steps=10))
                canvas = [((value + 63) // 64) * 64 for value in (width, height)]
                self.assertEqual((args.width, args.height, args.steps), (width, height, 10))
                self.assertEqual(args.krea2['output_size'], [width, height])
                self.assertEqual(args.krea2['canvas_size'], canvas)
                self.assertAlmostEqual(args.native_flow_shift, default_mu('raw', *canvas))
                _, replay = standalone_image.resolve_arguments(standalone_image.build_parser().parse_values(
                    native_image.configuration_values(args)))
                self.assertEqual((replay.width, replay.height), (width, height))
                self.assertEqual(replay.native_sigmas, args.native_sigmas)

    def test_tensor_signatures_and_component_contracts_for_every_added_family(self):
        for family, shapes in SIGNATURES.items():
            for prefix in ("", "model.diffusion_model.", "transformer."):
                with self.subTest(family=family, prefix=prefix):
                    path = self.safetensors({prefix + k: v for k, v in shapes.items()})
                    report = downloaded_model.inspect_downloaded_model(path)
                    self.assertEqual(report["architecture"], family)
                    plan = report["backend_plan"]
                    self.assertEqual(plan["missing_components"], ["vae", *registry.PROFILES[family][0]])
                    self.assertFalse(plan["ready_for_loading"])

    def test_all_families_reach_component_aware_runner_and_replay(self):
        for family in SIGNATURES:
            with self.subTest(family=family):
                # A valid header is routing evidence; only a real context load
                # can check the tensor contents represented by these fixtures.
                args = self.request(family)
                with patch.object(standalone_image, "inspect_checkpoint", side_effect=AssertionError("preset fallback")):
                    _, resolved = standalone_image.resolve_arguments(args)
                self.assertEqual((resolved.engine, resolved.architecture, resolved.native_v2), ("native", family, True))
                self.assertTrue(resolved.native_plan["ready_for_loading"])
                replay = native_image.configuration_values(resolved)
                _, copied = standalone_image.resolve_arguments(standalone_image.build_parser().parse_values(replay))
                self.assertEqual(copied.components, resolved.components)
                self.assertEqual(copied.guidance_scale, resolved.guidance_scale)

    def test_krea2_comfy_embedded_qwen3vl_is_not_reported_missing(self):
        shapes = dict(SIGNATURES['krea2'])
        shapes.update({'text_encoders.qwen3vl_4b.transformer.model.embed_tokens.weight': [2, 2],
                       'vae.encoder.conv1.weight': [1]})
        report = downloaded_model.inspect_downloaded_model(self.safetensors(shapes))
        self.assertEqual(report['component_slots'], ['llm', 'vae'])
        self.assertEqual(report['backend_plan']['missing_components'], [])
        foreign = dict(shapes)
        foreign.pop('text_encoders.qwen3vl_4b.transformer.model.embed_tokens.weight')
        foreign['text_encoders.qwen3vl_4b_backup.transformer.model.embed_tokens.weight'] = [2, 2]
        self.assertEqual(downloaded_model.inspect_downloaded_model(self.safetensors(foreign))
                         ['backend_plan']['missing_components'], ['llm'])

    def test_incompatible_component_slots_and_metadata_do_not_load(self):
        args = self.request("z-image")
        args.components["clip_l"] = self.component("clip_l")
        with self.assertRaisesRegex(ValueError, "do not belong"):
            standalone_image.resolve_arguments(args)
        for override in ({"base_model": "SD 1.5"}, {"guidance_scale": float("nan")}, {"engine": "diffusers"}):
            with self.subTest(override=override), self.assertRaises((ValueError, SystemExit)), patch("sys.stderr", io.StringIO()):
                standalone_image.resolve_arguments(self.request("z-image", **override))

    def test_sd_prediction_override_is_preserved_but_not_applied_to_flow_models(self):
        _, args = standalone_image.resolve_arguments(self.request("sd2", prediction_type="v_prediction"))
        self.assertEqual(args.prediction_type, "v_prediction")
        self.assertEqual(native_image.configuration_values(args)["prediction_type"], "v_prediction")
        with self.assertRaisesRegex(ValueError, "Flow backends"):
            standalone_image.resolve_arguments(self.request("z-image", prediction_type="epsilon"))

    def test_partial_sdxl_encoder_does_not_count_as_two_encoders(self):
        model = self.safetensors({"input_blocks.0.0.weight": [1, 4, 1, 1],
            "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight": [1, 2048],
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": [1],
            "first_stage_model.encoder.conv_in.weight": [1]})
        report = downloaded_model.inspect_downloaded_model(model)
        self.assertEqual(report["backend_plan"]["missing_components"], ["clip_g"])

    def test_krea_uses_flux1_and_turbo_uses_metadata_not_filename(self):
        args = self.request("flux1-dev", base_model="Flux.1 Krea")
        _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(resolved.architecture, "flux1-dev")
        args = self.request("z-image", base_model="ZImageTurbo")
        _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual((resolved.steps, resolved.guidance_scale), (9, 1.0))

    def test_layered_rgba_is_not_routed_to_rgb_decoder(self):
        shapes = dict(SIGNATURES["qwen-image"], **{"time_text_embed.addition_t_embedding.weight": [1]})
        report = downloaded_model.inspect_downloaded_model(self.safetensors(shapes))
        self.assertEqual(report["architecture"], "qwen-image-layered")
        with self.assertRaisesRegex(ValueError, "No native image backend"):
            registry.plan_native(report)

    def test_directory_inspection_routes_configured_pipelines_without_executing_code(self):
        for name, family in (("Flux2KleinPipeline", "flux2-klein"), ("ZImagePipeline", "z-image"),
                             ("Krea2Pipeline", "krea2"), ("WanPipeline", "wan")):
            (self.directory / "model_index.json").write_text(json.dumps({"_class_name": name,
                "transformer": ["diffusers", "SomeTransformer"]}))
            report = registry.inspect_pipeline_package(self.directory)
            self.assertEqual((report["architecture"], report["backend"], report["confidence"]),
                             (family, "diffusers", "configuration"))
        (self.directory / "model_index.json").write_text('{"_class_name": "../../run.py"}')
        with self.assertRaises(ValueError):
            registry.inspect_pipeline_package(self.directory)

    def test_quantized_container_route_preserves_original_file(self):
        path = self.gguf(architecture="z_image", name="quantized.gguf")
        report = downloaded_model.inspect_downloaded_model(path)
        self.assertEqual(report["architecture"], "z-image")
        args = self.request("z-image", model=str(path))
        _, resolved = standalone_image.resolve_arguments(args)
        self.assertEqual(resolved.model, str(path.resolve()))
        self.assertEqual(resolved.native_plan["architecture"], "z-image")

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow required")
    def test_changed_companion_during_generation_prevents_publication(self):
        from PIL import Image
        args = self.request("z-image")
        output = self.directory / "failed-output"
        class Engine:
            def image(self, request, seed, **kwargs):
                Path(request.components["llm"]).write_bytes(b"changed")
                return Image.new("RGB", (64, 64)), {}
        with patch.object(native_image, "NativeEngine", Engine), self.assertRaisesRegex(SystemExit, "changed"):
            standalone_image.main(["--model", args.model, "--components", json.dumps(args.components),
                "--width", "64", "--height", "64", "--output-dir", str(output)])
        self.assertEqual(list(output.iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow required")
    def test_native_v2_ctypes_boundary_carries_all_components_and_validates_rgb(self):
        _, args = standalone_image.resolve_arguments(self.request("flux1-dev", device="cpu"))
        engine = object.__new__(native_image.NativeEngine)
        received = []
        @C.CFUNCTYPE(C.c_void_p, C.POINTER(native_image.RequestV2), native_image.Progress, native_image.Preview, C.c_void_p)
        def call(pointer, progress, preview, user):
            value = pointer.contents
            received.append((value.clip_l.decode(), value.t5xxl.decode(), value.vae.decode(), value.cpu,
                             value.guidance_scale, value.distilled_guidance, value.hires, value.image.seed))
            return 123
        from types import SimpleNamespace
        engine.library = SimpleNamespace(iild_native_generate_v2=call)
        engine.metadata = lambda handle: json.dumps({"error": "", "cancelled": False, "width": 64, "height": 64})
        buffer = C.create_string_buffer(bytes((20, 40, 80)) * 64 * 64)
        def rgb(handle, size):
            C.cast(size, C.POINTER(C.c_size_t))[0] = 64 * 64 * 3
            return C.addressof(buffer)
        engine.rgb, engine.free = rgb, lambda handle: None
        image, _ = engine.image(args, 71)
        self.assertEqual(image.getpixel((0, 0)), (20, 40, 80))
        self.assertEqual(received[0], (args.components["clip_l"], args.components["t5xxl"], args.components["vae"], 1, 1.0, 3.5, 0, 71))

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow required')
    def test_native_v3_ctypes_boundary_carries_sampling(self):
        _, args = standalone_image.resolve_arguments(self.request('krea2', krea2_variant='turbo',
            sigmas=[1, .5], krea2_mu=0.0))
        engine = object.__new__(native_image.NativeEngine)
        received = []
        @C.CFUNCTYPE(C.c_void_p, C.POINTER(native_image.RequestV3), native_image.Progress, native_image.Preview, C.c_void_p)
        def call(pointer, progress, preview, user):
            value = pointer.contents
            received.append((value.size, value.image.size, value.sampler, value.flow_shift,
                             list(value.sigmas[:value.sigma_count]), value.image.image.steps))
            return 123
        from types import SimpleNamespace
        engine.library = SimpleNamespace(iild_native_generate_v2=SimpleNamespace(), iild_native_generate_v3=call)
        engine.metadata = lambda handle: json.dumps({'error': '', 'cancelled': False, 'width': 64, 'height': 64})
        buffer = C.create_string_buffer(bytes((20, 40, 80)) * 64 * 64)
        def rgb(handle, size):
            C.cast(size, C.POINTER(C.c_size_t))[0] = 64 * 64 * 3
            return C.addressof(buffer)
        engine.rgb, engine.free = rgb, lambda handle: None
        image, _ = engine.image(args, 71)
        self.assertEqual(received, [(C.sizeof(native_image.RequestV3), C.sizeof(native_image.RequestV2),
                                    1, 0.0, [1, .5, 0], 2)])

    def test_public_inspector_exposes_backend_plan_without_loading_engine(self):
        spec = importlib.util.spec_from_file_location("architecture_router", ROOT / "reference/generate.py")
        router = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(router)
        args = self.request("z-image")
        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(native_image, "NativeEngine", side_effect=AssertionError("loaded engine")):
            router.main(["--inspect-model", "--model", args.model, "--components", json.dumps(args.components)])
        self.assertTrue(json.loads(output.getvalue())["backend_plan"]["ready_for_loading"])
        request = self.directory / "request.json"
        request.write_text(json.dumps({"model": args.model, "components": args.components,
                                      "engine": "native"}))
        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(native_image, "NativeEngine", side_effect=AssertionError("loaded engine")):
            self.assertEqual(router.main(["--config", str(request), "--validate-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["engine"], "native")


if __name__ == "__main__":
    unittest.main()
