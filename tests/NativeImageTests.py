#!/usr/bin/env python3
"""Anima worker routing, retained model preparation and transactional queue output."""
import io
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import DownloadedModelTests as fixtures
import native_image
import standalone_image
from inference_worker import serve


class NativeImageTests(unittest.TestCase):
    setUp = fixtures.DownloadedModelTests.setUp
    safetensors = fixtures.DownloadedModelTests.safetensors

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

    def test_missing_components_and_unimplemented_overrides_are_rejected(self):
        incomplete = self.safetensors({"net.llm_adapter.blocks.0.cross_attn.q_proj.weight": [2, 2]})
        with self.assertRaisesRegex(ValueError, "missing components: vae, text_encoder"):
            standalone_image.resolve_arguments(self.values(incomplete))
        for values in ({"vae": "different.safetensors"}, {"guidance_scale": 7.5},
                       {"device": "cpu"}, {"base_model": "SD 1.5"},
                       {"width": 65}, {"num_images": 0}, {"seed": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                standalone_image.resolve_arguments(self.values(**values))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for real PNG validation")
    def test_worker_prepares_without_sampling_then_completes_ten_requests(self):
        from PIL import Image
        models, prepared, seeds = [], [], []
        class Engine:
            def __init__(self): models.append(self)
            def image(self, args, seed, *, prepare=False):
                if prepare:
                    prepared.append(True)
                    return None, {}
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
        self.assertEqual((len(models), len(prepared), seeds), (1, 1, list(range(10))))
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
                calls.append(seed)
                if len(calls) == 2: raise RuntimeError("decoder failure")
                return Image.new("RGB", (args.width, args.height)), {}
        output = self.directory / "failed"
        with patch.object(native_image, "NativeEngine", Engine), self.assertRaisesRegex(SystemExit, "decoder failure"):
            standalone_image.main(["--model", str(self.model()), "--width", "64", "--height", "120",
                                   "--num-images", "10", "--output-dir", str(output)])
        self.assertEqual(list(output.iterdir()), [])

    def test_installed_library_is_resolved_from_its_own_prefix(self):
        prefix = self.directory / "installed"
        filename = "libiiLocalDiffusion.dylib"
        path = prefix / "lib" / filename
        path.parent.mkdir(parents=True)
        path.touch()
        with patch.object(native_image, "__file__", str(prefix / "share/iiLocalDiffusion/reference/diffusers/native_image.py")), \
                patch.object(native_image.sys, "platform", "darwin"), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(native_image.library_path(), path)


if __name__ == "__main__":
    unittest.main()
