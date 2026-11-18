#!/usr/bin/env python3
"""Anima worker routing, retained model preparation and transactional queue output."""
import io
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import DownloadedModelTests as fixtures
import native_image
import standalone_image
from inference_worker import serve


class NativeImageTests(unittest.TestCase):
    setUp = fixtures.DownloadedModelTests.setUp
    safetensors = fixtures.DownloadedModelTests.safetensors

    def test_macos_loader_rejects_a_different_library_selected_by_dyld(self):
        requested = self.directory / "current" / "libiiLocalDiffusion.dylib"
        actual = self.directory / "old" / "libiiLocalDiffusion.dylib"
        for file in (requested, actual):
            file.parent.mkdir()
            file.touch()
        with patch.object(native_image.sys, "platform", "darwin"), \
             patch.object(native_image.C, "CDLL", return_value=Mock()), \
             patch.object(native_image, "loaded_library_path", return_value=actual, create=True):
            with self.assertRaisesRegex(RuntimeError, "DYLD_LIBRARY_PATH") as failure:
                native_image.load_library(requested)
        self.assertIn(str(requested), str(failure.exception))
        self.assertIn(str(actual), str(failure.exception))

    def test_macos_loader_accepts_symlinks_to_the_requested_library(self):
        requested = self.directory / "libiiLocalDiffusion.dylib"
        requested.touch()
        alias = self.directory / "alias.dylib"
        alias.symlink_to(requested)
        library = Mock()
        with patch.object(native_image.sys, "platform", "darwin"), \
             patch.object(native_image.C, "CDLL", return_value=library), \
             patch.object(native_image, "loaded_library_path", return_value=alias, create=True):
            self.assertIs(native_image.load_library(requested), library)

    def test_legacy_bridge_rejects_an_explicit_hires_disable(self):
        engine = native_image.NativeEngine.__new__(native_image.NativeEngine)
        with self.assertRaisesRegex(ValueError, "component-aware"):
            engine.image(SimpleNamespace(native_v2=False, hires_fix=False), 17)

    def test_hires_provider_passes_sampling_and_model_options_to_native_bridge(self):
        from native_upscaler import NativeUpscaler, Options
        upscaler = NativeUpscaler.__new__(NativeUpscaler)
        upscaler.kind, upscaler.model, upscaler.device = 6, '/models/swin.safetensors', 'metal'
        upscaler.tile_size, upscaler.threads = 8, 3
        upscaler.max_weight_bytes, upscaler.max_working_bytes = 1234567, 7654321
        upscaler.srvgg_activation, upscaler.swinir_image_range, upscaler.swinir_qk_scale = 0, 255.0, .25
        args = SimpleNamespace(progress=False, native_loras=[], model='/models/base.safetensors', prompt='hello',
            negative_prompt='', generation_resources=None, width=64, height=64, steps=2, default_modifiers=False,
            native_v2=True, native_v3=True, components={}, guidance_scale=1.5, embedded_guidance=3.5,
            device='cpu', prediction_type='auto', native_sampler='heun', native_flow_shift=.8,
            native_sigmas=[1.0, .5, 0.0])
        engine = native_image.NativeEngine.__new__(native_image.NativeEngine)
        engine.library = Mock()
        seen = {}
        def capture(pointer, *_):
            value = C.cast(pointer, C.POINTER(native_image.RequestHires)).contents
            selected = value.upscaler.upscaler.image
            seen.update(hires=value.image.image.hires, sampler=value.image.sampler,
                sigmas=[value.image.sigmas[i] for i in range(value.image.sigma_count)],
                kind=selected.kind, model=selected.model, passes=selected.passes,
                threads=selected.threads, device=selected.cpu, tile=selected.tile_size,
                limit=value.upscaler.upscaler.max_working_bytes,
                image_range=value.upscaler.swinir_image_range, qk=value.upscaler.swinir_qk_scale,
                denoise=value.denoise_strength)
            return 1
        engine.library.iild_native_generate_hires_v1.side_effect = capture
        engine.metadata = lambda _: b'{"error":"captured","cancelled":false}'
        engine.free = Mock()
        with self.assertRaisesRegex(RuntimeError, 'captured'):
            engine.image(args, 17, hires_upscaler=upscaler, hires_options=Options(passes=2),
                         hires_denoise_strength=.4)
        self.assertEqual((seen['hires'], seen['sampler'], seen['sigmas']), (1, 2, [1.0, .5, 0.0]))
        self.assertEqual((seen['kind'], seen['model'], seen['passes'], seen['threads'], seen['limit']),
                         (6, b'/models/swin.safetensors', 2, 3, 7654321))
        self.assertEqual((seen['device'], seen['tile']), (2, 8))
        self.assertEqual((seen['image_range'], seen['qk']), (255.0, .25))
        self.assertAlmostEqual(seen['denoise'], .4)
        engine.free.assert_called_once_with(1)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required")
    def test_worker_uses_explicit_hires_filter_and_strength(self):
        from PIL import Image
        calls = []
        class Engine:
            library = SimpleNamespace(_name="fixture-library")
            def image(self, args, seed, *, prepare=False, **options):
                if prepare:
                    return None, {"weight_storage": "anonymous"}
                calls.append(options)
                return Image.new("RGB", (args.width, args.height)), {}
        engine = Engine()
        output = self.directory / "explicit-hires"
        with patch.object(native_image, "cached_pipeline", return_value=(engine, True)), \
             patch.object(native_image, "NativeUpscaler") as provider, \
             patch.object(native_image, "verify_pipeline_sources"):
            self.assertEqual(standalone_image.main(["--model", str(self.model()), "--width", "64", "--height", "120",
                "--hires-fix", "--hires-upscaler", "bicubic", "--hires-strength", "0.4", "--output-dir", str(output)]), 0)
        provider.assert_called_once_with("bicubic", library="fixture-library")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["hires_options"].target_width, calls[0]["hires_options"].target_height), (64, 128))
        self.assertEqual(calls[0]["hires_denoise_strength"], .4)
        manifest = json.loads((output / "generation.json").read_text())
        self.assertTrue(manifest["images"][0]["fixture"]["hires"])

    def test_adapter_collection_does_not_release_runtime_sources(self):
        import gc
        from inference_session import InferenceSession
        library = Mock(_name="native-fixture")
        library.iild_native_available_v1.return_value = 1
        with patch.object(native_image, "library_path", return_value=Path("fixture")), \
                patch.object(native_image.C, "CDLL", return_value=library), \
                patch.object(native_image, "loaded_library_path", return_value=None, create=True):
            with InferenceSession():
                engine = native_image.NativeEngine()
                del engine
                gc.collect()
                library.iild_native_release_v1.assert_not_called()
            library.iild_native_release_v1.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required")
    def test_native_previews_survive_refinement_step_restart(self):
        from PIL import Image
        directory = self.directory / "native-preview"
        writer = native_image.NativePreviewWriter(directory)
        output = io.StringIO()
        with patch("sys.stdout", output):
            writer(10, 10, 2, 2, bytes((12, 34, 56)) * 4)
            writer(1, 3, 2, 2, bytes((98, 76, 54)) * 4)
        events = [json.loads(line.removeprefix("IILD_PREVIEW ")) for line in output.getvalue().splitlines()]
        self.assertEqual([e["sequence"] for e in events], [1, 2])
        self.assertEqual([e["total_steps"] for e in events], [10, 3])
        self.assertEqual(Image.open(directory / events[0]["image"]).getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(Image.open(directory / events[1]["image"]).getpixel((0, 0)), (98, 76, 54))
        with self.assertRaises(ValueError): writer(4, 3, 2, 2, b"bad")
        with self.assertRaises(ValueError): native_image.NativePreviewWriter(directory)

    def test_publication_trace_survives_failure_without_claiming_completion(self):
        trace = self.directory / "trace.jsonl"
        trace.write_text("")
        performance = {"telemetry_path": str(trace), "telemetry_elapsed_ms": 2000}
        with patch("sys.stdout", io.StringIO()), patch.object(native_image.time, "monotonic", return_value=3):
            native_image.publication_event(performance, "publication-failed", 2, "disk full")
        record = json.loads(trace.read_text())
        self.assertEqual(record["event"], "publication-failed")
        self.assertEqual(record["phase"], "postprocess")
        self.assertEqual(record["elapsed_ms"], 3000)
        self.assertEqual(record["detail"], "disk full")

    def model(self, **kwargs):
        return self.safetensors({
            "model.diffusion_model.net.llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2],
            "model.diffusion_model.net.blocks.0.self_attn.q_proj.weight": [2, 2],
            "vae.encoder.conv1.weight": [1], "vae.decoder.conv1.weight": [1],
            "text_encoders.llm.model.embed_tokens.weight": [2, 2],
        }, **kwargs)

    def values(self, model=None, **kwargs):
        return standalone_image.build_parser().parse_values({
            "model": str(model or self.model()), "width": 64, "height": 120,
            "steps": 10, "prompt": "a portrait", **kwargs})

    def test_complete_anima_routes_before_diffusers_or_vae_fallback(self):
        with patch.object(standalone_image, "inspect_checkpoint", side_effect=AssertionError("Diffusers routing")):
            preset, args = standalone_image.resolve_arguments(self.values())
        self.assertIsNone(preset)
        self.assertEqual(args.engine, "native")
        self.assertIsNone(args.vae_file)
        self.assertEqual((args.width, args.height, args.steps), (64, 120, 10))
        self.assertEqual(args.negative_prompt, "")

    def test_native_anima_routes_safetensors_with_a_nonstandard_extension(self):
        for name in ("anima.bin", "anima.data", "anima.gguf", "anima.json"):
            with self.subTest(name=name):
                model = self.model(name=name)
                preset, args = standalone_image.resolve_arguments(self.values(model))
                self.assertIsNone(preset)
                self.assertEqual((args.engine, args.architecture), ("native", "anima"))
                self.assertEqual(Path(args.model), model)

    def test_split_anima_automatically_mounts_verified_catalog_components(self):
        model = self.safetensors({"net.llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2],
                                  "net.blocks.0.self_attn.q_proj.weight": [2, 2]}, name="split.safetensors")
        llm = self.safetensors({"fixture.weight": [1]}, name="anima-llm.safetensors")
        vae = self.safetensors({"fixture.weight": [1]}, name="anima-vae.safetensors")
        def entry(path, **extra):
            return {"file": path.name, "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **extra}
        catalog = {"version": 1,
            "fallback_llms": [entry(llm, families=["anima"], class_name="Qwen3ForCausalLM")],
            "fallback_vae": entry(vae, families=["qwen-image"], class_name="AutoencoderKLQwenImage")}
        (self.directory / "generation-defaults.json").write_text(json.dumps(catalog))
        _, resolved = standalone_image.resolve_arguments(self.values(model, generation_resources=str(self.directory),
                                                                     default_modifiers=False))
        self.assertEqual(resolved.components, {"llm": str(llm), "vae": str(vae)})
        self.assertTrue(resolved.native_plan["ready_for_loading"])
        _, explicit = standalone_image.resolve_arguments(self.values(model, generation_resources=str(self.directory),
                                                                     components={"llm": str(llm)}))
        self.assertEqual(explicit.components, {"llm": str(llm), "vae": str(vae)})
        catalog["fallback_llms"][0]["class_name"] = "Qwen3VLForConditionalGeneration"
        (self.directory / "generation-defaults.json").write_text(json.dumps(catalog))
        with self.assertRaisesRegex(ValueError, "Qwen3-0.6B"):
            standalone_image.resolve_arguments(self.values(model, generation_resources=str(self.directory)))

    def test_missing_components_and_unimplemented_overrides_are_rejected(self):
        incomplete = self.safetensors({"net.llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2]})
        (self.directory / "generation-defaults.json").write_text('{"version":1,"fallback_llms":[]}')
        with self.assertRaisesRegex(ValueError, "Qwen3-0.6B"):
            standalone_image.resolve_arguments(self.values(incomplete, generation_resources=str(self.directory)))
        for values in ({"vae": "different.safetensors"}, {"guidance_scale": -1},
                       {"device": "cuda"}, {"base_model": "SD 1.5"},
                       {"width": 65}, {"num_images": 0}, {"seed": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                standalone_image.resolve_arguments(self.values(**values))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for real PNG validation")
    def test_worker_prepares_without_sampling_then_completes_ten_requests(self):
        from PIL import Image
        models, prepared, seeds, events = [], [], [], []
        class Engine:
            def __init__(self): models.append(self)
            def image(self, args, seed, *, prepare=False):
                if prepare:
                    prepared.append(True)
                    events.append("load")
                    return None, {"weight_storage": "anonymous", "model_cache_hit": len(prepared) > 1}
                events.append("generate")
                seeds.append(seed)
                return Image.new("RGB", (args.width, args.height), (seed, 80, 120)), {"model_cache_hit": True}
        model = self.model()
        common = ["--model", str(model), "--width", "64", "--height", "120", "--steps", "10"]
        requests = [{"id": "prepare", "action": "foreground", "foreground": True, "arguments": common}]
        for index in range(10):
            requests.append({"id": str(index), "arguments": common + ["--prompt", "portrait", "--seed", str(index),
                             "--output-dir", str(self.directory / str(index)),
                             "--work-dir", str(self.directory / (str(index) + "-work")),
                             "--cache-dir", str(self.directory / (str(index) + "-cache")),
                             "--preview-dir", str(self.directory / (str(index) + "-preview")), "--device", "auto"]})
        stream = io.BytesIO("".join(json.dumps({"schema": "iild-worker-request-v1", **r}) + "\n" for r in requests).encode())
        output = io.StringIO()
        with patch.object(native_image, "NativeEngine", Engine), patch("sys.stdout", output):
            self.assertEqual(serve(standalone_image.main, stream), 0)
        results = [json.loads(line.removeprefix("IILD_RESULT ")) for line in output.getvalue().splitlines() if line.startswith("IILD_RESULT ")]
        self.assertEqual(len(results), 11)
        self.assertTrue(all(result["ok"] for result in results), results)
        self.assertEqual((len(models), len(prepared), seeds), (1, 11, list(range(10))))
        self.assertEqual(events, ["load"] + ["load", "generate"] * 10)
        self.assertTrue(results[0]["residency"]["ready"])
        self.assertFalse(results[0]["residency"]["gpu_resident"])  # lazy placement, never claim all weights on GPU
        for index in range(10):
            manifest = json.loads((self.directory / str(index) / "generation.json").read_text())
            self.assertEqual(manifest["backend"], "native")
            self.assertEqual(manifest["images"][0]["fixture"]["seed"], index)
            self.assertEqual(manifest["images"][0]["vae"], {"source": "checkpoint", "override": None})
            self.assertEqual(Path(manifest["outputs"][0]["path"]).parent, self.directory / str(index))
            request = json.loads((self.directory / (str(index) + "-work") / "request.json").read_text())
            # A saved native request must not inherit unsupported Diffusers
            # defaults, and must resolve again through the same native route.
            _, replay = standalone_image.resolve_arguments(standalone_image.build_parser().parse_values(request))
            self.assertEqual((replay.engine, replay.width, replay.height, replay.steps), ("native", 64, 120, 10))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for real PNG validation")
    def test_failed_batch_does_not_publish_partial_images(self):
        from PIL import Image
        calls = []
        class Engine:
            def image(self, args, seed, **kwargs):
                if kwargs.get("prepare"):
                    return None, {"weight_storage": "anonymous"}
                calls.append(seed)
                if len(calls) == 2: raise RuntimeError("decoder failure")
                return Image.new("RGB", (args.width, args.height)), {}
        output = self.directory / "failed"
        with patch.object(native_image, "NativeEngine", Engine), self.assertRaisesRegex(SystemExit, "decoder failure"):
            standalone_image.main(["--model", str(self.model()), "--width", "64", "--height", "120",
                                   "--num-images", "10", "--output-dir", str(output)])
        self.assertFalse(output.exists(), "A failed private batch must not publish its output directory")

    def test_queue_refuses_generation_until_anonymous_preload_succeeds(self):
        for metadata in ({}, {"weight_storage": "file-backed"}):
            calls = []
            class Engine:
                def image(self, args, seed, *, prepare=False):
                    calls.append(prepare)
                    if not prepare:
                        raise AssertionError("Inference started before resident preload")
                    return None, metadata
            with self.subTest(metadata=metadata), patch.object(native_image, "NativeEngine", Engine), \
                 self.assertRaisesRegex(SystemExit, "anonymous memory"):
                standalone_image.main(["--model", str(self.model()), "--width", "64", "--height", "120",
                                       "--output-dir", str(self.directory / ("rejected-" + str(len(metadata))))])
            self.assertEqual(calls, [True])

    def test_installed_library_is_resolved_from_its_own_prefix(self):
        prefix = self.directory / "installed"
        filename = "libiiLocalDiffusion.dll" if os.name == "nt" else "libiiLocalDiffusion.dylib"
        path = prefix / ("bin" if os.name == "nt" else "lib") / filename
        path.parent.mkdir(parents=True)
        path.touch()
        with patch.object(native_image, "__file__", str(prefix / "share/iiLocalDiffusion/reference/diffusers/native_image.py")), \
                patch.object(native_image.sys, "platform", "darwin"), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(native_image.library_path(), path)


if __name__ == "__main__":
    unittest.main()
