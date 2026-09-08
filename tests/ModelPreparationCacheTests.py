#!/usr/bin/env python3
"""Model construction and device placement outlive individual generation settings."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
import inference_session


class Scheduler:
    def __init__(self, prediction_type="epsilon", beta_start=0.00085):
        self.config = dict(prediction_type=prediction_type, beta_start=beta_start)
        self.compatibles = [Scheduler, AlternateScheduler]
        self.history = []

    @classmethod
    def from_config(cls, config, **values):
        return cls(**{**config, **values})

    def set_timesteps(self, steps, timesteps=None, sigmas=None):
        pass

    def step(self, *args, eta=0):
        pass


class AlternateScheduler(Scheduler):
    pass


class Pipeline:
    def __init__(self):
        self.scheduler = Scheduler()
        self.placements = []
        self.hooks_removed = 0
        self.attention = None
        self.vae = SimpleNamespace(enable_slicing=lambda: None, disable_slicing=lambda: None,
                                   enable_tiling=lambda: None, disable_tiling=lambda: None)
        self.text_encoder = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=12))

    def to(self, device):
        self.placements.append(("resident", device))
        return self

    def enable_model_cpu_offload(self, **args):
        self.placements.append(("model", args["device"]))

    def enable_sequential_cpu_offload(self, **args):
        self.placements.append(("sequential", args["device"]))

    def remove_all_hooks(self):
        self.hooks_removed += 1

    def enable_attention_slicing(self, value="auto"):
        self.attention = value

    def disable_attention_slicing(self):
        self.attention = None


class ModelPreparationCacheTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="model-preparation-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.folder = self.directory / "model"
        shutil.copytree(ROOT / "tests/fixtures/sd-v1-manifest", self.folder)
        stack = self.enterContext(ExitStack())
        self.session = stack.enter_context(inference_session.InferenceSession())
        self.load = stack.enter_context(patch.object(generate, "load_generation_pipeline",
                                                    side_effect=lambda *args: (Pipeline(), {})))
        stack.enter_context(patch.object(generate, "validate_pipeline_contract"))

    def prepare(self, device="cpu", **values):
        preset, args = generate.resolve_request({"preset": "sd15", "model": str(self.folder), **values})
        dtype = "float32" if args.dtype == "auto" else args.dtype
        loading = generate.build_load_arguments(preset, args, dtype, device != "cpu")
        result = generate.prepared_pipeline(preset, args, {preset.pipeline_class: Pipeline}, loading,
                                            device, dtype, bool(args.attention_slicing), None)
        return result[0], args, result[-1]

    def test_sampling_options_reuse_composition_and_placement_without_scheduler_leak(self):
        first, _, hit = self.prepare()
        self.assertFalse(hit)
        first.scheduler.history.append("old denoising state")
        second, args, hit = self.prepare(scheduler="AlternateScheduler", scheduler_config={"beta_start": 0.002},
                                         prediction_type="v_prediction", clip_skip=2, guidance_rescale=0.5,
                                         prompt="new prompt", seed=17, width=64, steps=3)
        self.assertTrue(hit)
        self.assertIs(second, first)
        self.assertIsInstance(second.scheduler, AlternateScheduler)
        self.assertEqual(second.scheduler.config["prediction_type"], "v_prediction")
        self.assertTrue(args.device_placement_cache_hit)
        third, _, hit = self.prepare()
        self.assertTrue(hit)
        self.assertIs(type(third.scheduler), Scheduler)
        self.assertEqual(third.scheduler.config, Scheduler().config)
        self.assertEqual(third.scheduler.history, [])
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(third.placements, [("resident", "cpu")])

    def test_device_or_attention_change_repositions_existing_components_only_once(self):
        pipeline, _, _ = self.prepare(attention_slicing=True)
        self.assertEqual(pipeline.attention, "auto")
        changed, args, hit = self.prepare(device="mps", attention_slicing=False)
        self.assertIs(changed, pipeline)
        self.assertFalse(hit)
        self.assertTrue(args.model_configuration_cache_hit)
        self.assertFalse(args.device_placement_cache_hit)
        self.assertIsNone(changed.attention)
        _, _, hit = self.prepare(device="mps", attention_slicing=False)
        self.assertTrue(hit)
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(pipeline.placements, [("resident", "cpu"), ("resident", "mps")])

    def test_offload_layout_is_retained_and_removed_before_resident_transition(self):
        pipeline, _, _ = self.prepare(device="mps", offload="model")
        again, _, hit = self.prepare(device="mps", offload="model", prompt="second")
        self.assertTrue(hit)
        self.assertIs(pipeline, again)
        _, _, hit = self.prepare(device="mps", offload="none")
        self.assertFalse(hit)
        self.assertEqual(pipeline.hooks_removed, 1)
        self.assertEqual(pipeline.placements, [("model", "mps"), ("resident", "mps")])
        self.assertEqual(self.load.call_count, 1)

    def test_model_config_file_or_dtype_change_rebuilds_both_stages(self):
        first, _, _ = self.prepare()
        index = self.folder / "model_index.json"
        index.write_text(index.read_text() + "\n")
        second, _, hit = self.prepare()
        self.assertFalse(hit)
        self.assertIsNot(first, second)
        third, _, hit = self.prepare(dtype="float16")
        self.assertFalse(hit)
        self.assertIsNot(second, third)
        self.assertEqual(self.load.call_count, 3)
        self.assertEqual(third.placements, [("resident", "cpu")])

    def test_failed_replacement_evicts_the_partially_moved_pipeline(self):
        first, _, _ = self.prepare()
        with patch.object(first, "to", side_effect=RuntimeError("device unavailable")):
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                self.prepare(device="mps")
        self.assertIsNone(self.session.value)
        next_pipeline, _, hit = self.prepare()
        self.assertFalse(hit)
        self.assertIsNot(first, next_pipeline)

    def test_generic_directory_retains_composition_and_placement_independently(self):
        import generate_any
        args = SimpleNamespace(source_kind="directory", model=str(self.folder), model_config=None,
                               pipeline_class=None, base_model=None, offload="none")
        with (patch.object(generate_any, "load_pipeline", side_effect=lambda *args: (Pipeline(), {})) as load,
              patch.object(generate_any, "validate_base_model", return_value={}),
              patch.object(generate_any, "validate_execution_device")):
            first, _, hit = generate_any.prepared_pipeline(args, None, "float32", "cpu")
            self.assertFalse(hit)
            _, _, hit = generate_any.prepared_pipeline(args, None, "float32", "cpu")
            self.assertTrue(hit)
            changed, _, hit = generate_any.prepared_pipeline(args, None, "float32", "mps")
            self.assertFalse(hit)
            self.assertTrue(args.model_configuration_cache_hit)
            self.assertIs(changed, first)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first.placements, [("resident", "cpu"), ("resident", "mps")])
        self.assertEqual(self.session.statistics()["device_placements"], 2)
        self.assertEqual(self.session.statistics()["device_placement_hits"], 1)
        self.assertEqual(self.session.statistics()["configuration_reads"], 1)
        self.assertEqual(self.session.statistics()["configuration_hits"], 2)

    def test_checkpoint_inspection_cache_detects_sidecar_addition_change_and_removal(self):
        import DownloadedModelTests
        import downloaded_model
        import checkpoint_config
        fixture = DownloadedModelTests.DownloadedModelTests()
        fixture.directory = self.directory
        model = fixture.sd()
        with patch.object(downloaded_model, "_safetensors", wraps=downloaded_model._safetensors) as reader:
            first = downloaded_model.inspect_downloaded_model(model)
            first["evidence"].append("caller mutation")
            cached = downloaded_model.inspect_downloaded_model(model)
            checkpoint_config.bundled_configuration(model, generate.SD15_PRESET)
            self.assertNotIn("caller mutation", cached["evidence"])
            self.assertEqual(reader.call_count, 1)
            fixture.info(model)
            downloaded_model.inspect_downloaded_model(model)
            self.assertEqual(reader.call_count, 2)
            sidecar = model.with_suffix(".civitai.info")
            info = json.loads(sidecar.read_text())
            info["baseModel"] = "SDXL 1.0"
            sidecar.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "conflict"):
                downloaded_model.inspect_downloaded_model(model)
            sidecar.unlink()
            downloaded_model.inspect_downloaded_model(model)
            self.assertEqual(reader.call_count, 4)

    @unittest.skipUnless(os.environ.get("IILD_PLACEMENT_MODEL"), "Opt-in requires a local SD1 Diffusers model and MPS")
    def test_real_offload_and_cpu_mps_roundtrip_retains_the_model_and_pixels(self):
        import generate_any
        original = None
        results = []
        cases = [("cpu", "none"), ("mps", "none"), ("mps", "model"), ("mps", "model"),
                 ("mps", "sequential"), ("mps", "sequential"), ("mps", "none"), ("cpu", "none")]
        for index, (device, offload) in enumerate(cases):
            with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
                root = Path(temporary)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(generate_any.main([
                        "--model", os.environ["IILD_PLACEMENT_MODEL"], "--prompt", "a red cube", "--seed", "42",
                        "--width", "64", "--height", "64", "--steps", "1", "--dtype", "float32",
                        "--device", device, "--offload", offload,
                        "--output-dir", str(root / "output"), "--cache-dir", str(root / "cache"),
                        "--preview-dir", str(root / "preview"),
                    ]), 0)
                pipeline = self.session.value[0]
                if original is None:
                    original = pipeline
                self.assertIs(pipeline, original)
                report = json.loads((root / "output/generation.json").read_text())
                results.append(Path(report["outputs"][0]["path"]).read_bytes())
        self.assertEqual(self.session.loads, 1)
        self.assertEqual(self.session.device_placements, 6)
        self.assertEqual(self.session.configuration_reads, 1)
        self.assertEqual(results[0], results[7])
        self.assertEqual(results[2], results[3])
        self.assertEqual(results[4], results[5])


if __name__ == "__main__":
    unittest.main()
