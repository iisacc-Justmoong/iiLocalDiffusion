#!/usr/bin/env python3
"""Dependency-free contracts for two-pass generation and safe artifact publication."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference" / "diffusers"))
import generate
import generation_output
import hires
from hires_options import hires_request
import presets


class Image:
    def __init__(self, size=(32, 32), *, label="image", mode="RGB", extrema=None):
        self.size, self.label, self.mode = size, label, mode
        self.width, self.height = size
        self.extrema = ((0, 255), (1, 254), (2, 253)) if extrema is None else extrema
        self.resize_calls = []

    def convert(self, mode):
        return Image(self.size, label=self.label, mode=mode, extrema=self.extrema)

    def resize(self, size, resample):
        self.resize_calls.append((size, resample))
        return Image(size, label=self.label + ":upscaled", extrema=self.extrema)

    def getextrema(self):
        return self.extrema

    def tobytes(self):
        return f"{self.label}:{self.size}:{self.mode}".encode()

    def save(self, path, **options):
        Path(path).write_bytes(self.tobytes())


PIL = SimpleNamespace(Image=SimpleNamespace(Resampling=SimpleNamespace(
    NEAREST="nearest", BILINEAR="bilinear", BICUBIC="bicubic", LANCZOS="lanczos")))


class Timeline:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values[:]


class Scheduler:
    def __init__(self, *, events=None, label="base", config=None):
        self.events = [] if events is None else events
        self.label = label
        self.config = {"num_train_timesteps": 1000} if config is None else dict(config)
        self.timesteps = Timeline([900, 500, 100])
        self.compatibles = []

    @classmethod
    def from_config(cls, config, **options):
        return cls(label="copy", config={**config, **options})

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = Timeline(range(num_inference_steps))

    def step(self, prediction, timestep, latents):
        return latents


class Pipeline:
    def __init__(self, events, *, width=64, height=64, count=1, active=("custom",)):
        self.events = events
        self.scheduler = Scheduler(events=events)
        self.width, self.height, self.count = width, height, count
        self.hooks_removed = False
        self.on_cpu = False
        self.calls = []
        self.output = None
        self.invoke_callback = True
        self.finite = True
        self._execution_device = SimpleNamespace(type="cpu", index=None)
        self.unet = SimpleNamespace(active_adapters=lambda: self.adapter_names(active))
        self.transformer = self.unet
        self.text_encoder = self.unet
        self.vae = SimpleNamespace(config=SimpleNamespace(
            latent_channels=4, scaling_factor=0.18215, block_out_channels=(1, 2, 4, 4)))
        self.components = {}
        self.num_timesteps = 1

    def adapter_names(self, names):
        self.events.append("check-adapters")
        return names

    def remove_all_hooks(self):
        self.events.append("remove-hooks")
        self.hooks_removed = True

    def to(self, device):
        if not self.hooks_removed:
            raise RuntimeError("Offload hooks must restore the stored weights before moving the pipeline.")
        self.events.append(("move", device))
        self.on_cpu = device == "cpu"
        return self

    def set_progress_bar_config(self, **options):
        self.events.append(("progress", options))

    def __call__(self, **arguments):
        self.events.append("denoise")
        self.calls.append(arguments)
        if self.invoke_callback and "callback_on_step_end" in arguments:
            arguments["callback_on_step_end"](self, 0, 500, {"latents": SimpleNamespace(finite=self.finite)})
        if self.output is not None:
            return self.output
        return SimpleNamespace(images=[Image((self.width, self.height), label=f"render-{i}")
                                       for i in range(self.count)])


class Torch:
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self, events=None):
        self.events = [] if events is None else events
        self.generators = []

    def Generator(self, device):
        generator = SimpleNamespace(device=device)

        def seed(value):
            generator.seed = value
            self.events.append(("seed", value))
            return generator

        generator.manual_seed = seed
        self.generators.append(generator)
        return generator

    def isfinite(self, latents):
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: latents.finite))

    def inference_mode(self):
        return nullcontext()

    def get_num_threads(self):
        return 1


def request(preset=presets.SD15_PRESET, **values):
    data = {"preset": preset.name, "width": 32, "height": 32, "hires_fix": True,
            "device": "cpu", "hires_seed": 91, "hires_steps": 8}
    data.update(values)
    return generate.resolve_request(data)[1]


class RuntimeFixture:
    def __init__(self, preset=presets.SD15_PRESET, *, controlnet=False, **values):
        self.preset = preset
        self.args = request(preset, **values)
        if controlnet:
            self.args.controlnet_selection = object()
        self.events = []
        self.torch = Torch(self.events)
        self.base = Pipeline(self.events, width=32, height=32, count=self.args.num_images)
        self.refined = Pipeline(self.events, count=self.args.num_images)
        self.images = [Image(label=f"base-{i}") for i in range(self.args.num_images)]
        self.activation = None
        self.build_calls = []
        self.execution_calls = []
        self.conversion_calls = []

        def from_pipe(base, **options):
            if not base.hooks_removed or not base.on_cpu:
                raise AssertionError("Conversion must follow offload-hook removal and CPU restoration.")
            self.events.append("convert")
            self.conversion_calls.append((base, options))
            self.refined.scheduler = options["scheduler"]
            return self.refined

        self.classes = {hires.PIPELINES[(preset.name, controlnet)]: SimpleNamespace(from_pipe=from_pipe)}

    def build_call(self, preset, args, generator, conditioning, device, dtype):
        self.events.append("build-call")
        self.build_calls.append((args, generator, conditioning, device, dtype))
        result = {
            "width": args.width, "height": args.height, "num_inference_steps": args.steps,
            "num_images_per_prompt": args.num_images, "guidance_scale": args.guidance_scale,
            "generator": generator, "prompt": args.prompt,
            "latents": "stale-initial-latents", "sigmas": [1.0], "timesteps": [900],
            "denoising_end": 0.7, "guidance_rescale": 0.4,
        }
        if args.controlnet_selection is not None:
            result["controlnet_conditioning_scale"] = 0.7
            result["control_image" if preset.name == "flux1-schnell" else "image"] = "control-guide"
        return result

    def prepare(self, pipeline, preset, device, attention, **values):
        self.events.append("prepare-execution")
        self.execution_calls.append((pipeline, values))
        pipeline._execution_device = SimpleNamespace(type=device, index=values["device_index"])
        return pipeline, {"weight_storage": "cpu", "offload_policy": values["offload"]}

    def run(self, **overrides):
        with patch.dict(sys.modules, {"PIL": PIL}):
            return hires.run_hires_fix(
                self.base, self.preset, self.args, self.images, self.torch,
                overrides.pop("device", "cpu"), "selected-dtype", False, self.activation,
                build_call=self.build_call, prepare_execution=overrides.pop("prepare_execution", self.prepare),
                pipeline_classes=self.classes, **overrides)


class HiresRuntimeTests(unittest.TestCase):
    def test_upscaling_retains_batch_order_and_exposes_every_filter(self):
        for method in ("nearest", "bilinear", "bicubic", "lanczos"):
            originals = [Image(label="first"), Image(label="second")]
            with self.subTest(method=method), patch.dict(sys.modules, {"PIL": PIL}):
                result = hires.upscale_images(originals, 96, 64, method)
            self.assertEqual([image.label for image in result], ["first:upscaled", "second:upscaled"])
            self.assertTrue(all(image.size == (96, 64) for image in result))
            self.assertTrue(all(image.resize_calls == [((96, 64), method)] for image in originals))
        with patch.dict(sys.modules, {"PIL": PIL}), self.assertRaises(ValueError):
            hires.upscale_images([Image()], 64, 64, "unsupported")

    def test_all_six_call_types_keep_generated_and_control_images_distinct(self):
        for preset in presets.PRESETS.values():
            for controlnet in (False, True):
                with self.subTest(preset=preset.name, controlnet=controlnet):
                    fixture = RuntimeFixture(preset, controlnet=controlnet)
                    args = hires_request(preset, fixture.args)
                    original = fixture.build_call(preset, args, "new-seed", None, "cpu", "dtype")
                    images = [Image((64, 64), label="generated")]
                    result = hires.refinement_call_arguments(preset, args, original, images)
                    for name in ("latents", "timesteps", "sigmas", "denoising_end", "guidance_rescale"):
                        self.assertNotIn(name, result)
                        self.assertIn(name, original)
                    if controlnet and preset.name == "flux1-schnell":
                        self.assertEqual(result["control_image"], "control-guide")
                        self.assertNotIn("image", result)
                        self.assertNotIn("strength", result)
                    else:
                        self.assertIs(result["image"], images)
                        self.assertEqual(result["strength"], args.hires_denoising_strength)
                        if controlnet:
                            self.assertEqual(result["control_image"], "control-guide")
                    keeps_dimensions = controlnet or preset.name == "flux1-schnell"
                    self.assertEqual("width" in result, keeps_dimensions)
                    self.assertEqual("height" in result, keeps_dimensions)

    def test_runtime_passes_upscaled_first_images_and_reseeds_each_batch_member(self):
        fixture = RuntimeFixture(num_images=2, seed=3, seed_stride=7, hires_seed=101)
        fixture.args.latents_file = object()
        fixture.args.tensor_inputs = object()
        fixture.activation = SimpleNamespace(registered_components=("unet",), active_adapters=("custom",))
        result = fixture.run()
        call = fixture.refined.calls[0]
        self.assertEqual([image.label for image in call["image"]], ["base-0:upscaled", "base-1:upscaled"])
        self.assertTrue(all(image.size == (64, 64) for image in call["image"]))
        self.assertEqual([generator.seed for generator in call["generator"]], [101, 108])
        self.assertEqual(result.seeds, [101, 108])
        self.assertIsNone(result.request.latents_file)
        self.assertIsNone(result.request.tensor_inputs)
        self.assertIsNotNone(fixture.args.latents_file)
        self.assertNotIn("latents", call)
        self.assertEqual(result.metadata["refinement"]["executed_steps"], 1)
        self.assertTrue(result.metadata["refinement"]["finite_latents"])
        self.assertEqual(result.metadata["upscale"]["source_size"], [32, 32])
        self.assertEqual(result.metadata["upscale"]["target_size"], [64, 64])

    def test_offload_hooks_cpu_dtype_scheduler_and_adapters_survive_conversion(self):
        fixture = RuntimeFixture(presets.SDXL_BASE_PRESET, watermark=True)
        fixture.activation = SimpleNamespace(registered_components=("unet",), active_adapters=("custom",))
        original_scheduler = fixture.base.scheduler
        result = fixture.run()
        converted_base, options = fixture.conversion_calls[0]
        self.assertIs(converted_base, fixture.base)
        self.assertEqual(options["dtype"], "selected-dtype")
        self.assertTrue(options["add_watermarker"])
        self.assertIsNot(result.pipeline.scheduler, original_scheduler)
        self.assertEqual(result.pipeline.scheduler.config, original_scheduler.config)
        self.assertEqual(original_scheduler.timesteps.tolist(), [900, 500, 100])
        self.assertLess(fixture.events.index("remove-hooks"), fixture.events.index(("move", "cpu")))
        self.assertLess(fixture.events.index(("move", "cpu")), fixture.events.index("convert"))
        self.assertLess(fixture.events.index("convert"), fixture.events.index("check-adapters"))
        self.assertLess(fixture.events.index("check-adapters"), fixture.events.index("prepare-execution"))

    def test_lost_lora_activation_fails_before_denoising(self):
        fixture = RuntimeFixture()
        fixture.refined.unet = SimpleNamespace(active_adapters=lambda: ())
        fixture.activation = SimpleNamespace(registered_components=("unet",), active_adapters=("custom",))
        with self.assertRaisesRegex(RuntimeError, "LoRA"):
            fixture.run()
        self.assertEqual(fixture.refined.calls, [])

    def test_embeddings_are_rechecked_against_second_pass_guidance_before_execution(self):
        fixture = RuntimeFixture(guidance_scale=1.0, hires_guidance_scale=7.5)
        fixture.args.embeddings_file = object()
        previous_inputs = object()
        fixture.args.tensor_inputs = previous_inputs
        conditioning = SimpleNamespace(metadata={"enabled": False, "source": "embeddings"})

        def load(pipeline, preset, args, torch, dtype):
            fixture.events.append("reload-embeddings")
            self.assertEqual(args.guidance_scale, 7.5)
            self.assertIsNone(args.latents_file)
            self.assertIsNone(args.tensor_inputs)
            self.assertIs(args.embeddings_file, fixture.args.embeddings_file)
            return SimpleNamespace(conditioning=conditioning, latents=None)

        with patch.object(hires, "load_tensor_inputs", side_effect=load) as loader:
            result = fixture.run()
        loader.assert_called_once()
        self.assertIs(result.conditioning, conditioning)
        self.assertIs(fixture.build_calls[0][2], conditioning)
        self.assertIs(fixture.args.tensor_inputs, previous_inputs)
        self.assertLess(fixture.events.index("reload-embeddings"), fixture.events.index("prepare-execution"))

    def test_missing_second_pass_negative_embeddings_fail_before_execution(self):
        fixture = RuntimeFixture(guidance_scale=1.0, hires_guidance_scale=7.5)
        fixture.args.embeddings_file = object()
        with patch.object(hires, "load_tensor_inputs", side_effect=ValueError("negative embeddings required")):
            with self.assertRaisesRegex(ValueError, "negative embeddings"):
                fixture.run()
        self.assertEqual(fixture.execution_calls, [])
        self.assertEqual(fixture.refined.calls, [])

    def test_cpu_conditioning_precedes_offload_hooks_and_preserves_requested_gpu(self):
        fixture = RuntimeFixture(cpu_text_encoding=True, device="cuda", device_index=2)
        conditioning = SimpleNamespace(metadata={"enabled": True})

        def encode(*arguments):
            fixture.events.append("cpu-encode")
            self.assertNotIn("prepare-execution", fixture.events)
            return conditioning

        with patch.object(hires, "encode_cpu_prompt", side_effect=encode):
            result = fixture.run(device="cuda")
        self.assertEqual(fixture.execution_calls[0][1]["offload"], "model")
        self.assertEqual(fixture.execution_calls[0][1]["device_index"], 2)
        self.assertEqual(result.pipeline._execution_device.index, 2)
        self.assertIs(result.conditioning, conditioning)

    def test_wrong_execution_device_is_rejected_before_sampling(self):
        fixture = RuntimeFixture(device="cuda", device_index=1)

        def wrong_index(*arguments, **options):
            pipeline, optimization = fixture.prepare(*arguments, **options)
            pipeline._execution_device.index = 0
            return pipeline, optimization

        with self.assertRaisesRegex(RuntimeError, "device index"):
            fixture.run(device="cuda", prepare_execution=wrong_index)
        self.assertEqual(fixture.refined.calls, [])

    def test_flux_controlnet_helper_receives_upscaled_images_and_independent_guidance(self):
        fixture = RuntimeFixture(presets.FLUX1_SCHNELL_PRESET, controlnet=True, hires_true_cfg_scale=2.0)

        def refine(pipeline, args, images, call, torch, generator, device, dtype):
            self.assertEqual([image.label for image in images], ["base-0:upscaled"])
            self.assertEqual(call["control_image"], "control-guide")
            self.assertNotIn("image", call)
            self.assertNotIn("latents", call)
            self.assertEqual(args.true_cfg_scale, 2.0)
            self.assertEqual(generator.seed, 91)
            return pipeline(**call), {"conditioning": "img2img-latents-with-controlnet-and-true-cfg"}

        with patch("hires_flux_controlnet.refine_flux_controlnet", side_effect=refine) as helper:
            result = fixture.run()
        helper.assert_called_once()
        self.assertEqual(result.metadata["refinement"]["compatibility"]["conditioning"],
                         "img2img-latents-with-controlnet-and-true-cfg")

    def test_stage_validation_rejects_count_size_and_uniform_outputs(self):
        cases = [([], "Expected"), ([Image((32, 64))], "size"),
                 ([Image((64, 64), extrema=((1, 1), (2, 2), (3, 3)))], "uniform")]
        for images, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                hires.validate_stage_images(SimpleNamespace(images=images), 1, 64, 64)
        rgba = Image((64, 64), mode="RGBA")
        converted = hires.validate_stage_images(SimpleNamespace(images=[rgba]), 1, 64, 64)
        self.assertEqual(converted[0].mode, "RGB")
        self.assertEqual(rgba.mode, "RGBA")

    def test_denoising_audit_rejects_missing_nonfinite_and_empty_passes(self):
        audit = hires.DenoisingAudit(Torch())
        with self.assertRaisesRegex(RuntimeError, "non-empty"):
            audit.metadata()
        for values in ({}, {"latents": SimpleNamespace(finite=False)}):
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, "latents"):
                audit(None, 3, 1.0, values)
        values = {"latents": SimpleNamespace(finite=True), "extra": object()}
        self.assertIs(audit(None, 0, 12.5, values), values)
        self.assertIs(audit(None, 1, 0.0, values), values)
        self.assertEqual(audit.metadata(), {"executed_steps": 2, "executed_timesteps": [12.5, 0.0],
                                           "finite_latents": True})

    def test_runtime_cannot_report_success_without_real_sampling_or_valid_output(self):
        for mode in ("empty-sampling", "nonfinite", "bad-size", "bad-count", "uniform"):
            fixture = RuntimeFixture()
            if mode == "empty-sampling":
                fixture.refined.invoke_callback = False
            elif mode == "nonfinite":
                fixture.refined.finite = False
            else:
                images = [] if mode == "bad-count" else [Image(
                    (1, 1) if mode == "bad-size" else (64, 64),
                    extrema=((0, 0), (0, 0), (0, 0)) if mode == "uniform" else None)]
                fixture.refined.output = SimpleNamespace(images=images)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                fixture.run()

    def test_base_image_and_sidecar_collisions_allocate_a_new_default_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            original = root / "image-base.png"
            original.write_bytes(b"base-original")
            occupied = root / "image-run-0002-base.json"
            occupied.write_text("{}")
            args = request(hires_save_base=True)
            args.output = root / "image.png"
            outputs = generation_output.resolve_output_paths(args)
            self.assertEqual(outputs, [root / "image-run-0003.png"])
            self.assertEqual(generation_output.hires_base_paths(outputs), [root / "image-run-0003-base.png"])
            self.assertEqual(original.read_bytes(), b"base-original")

    def test_explicit_base_collisions_are_protected_and_overwrite_requires_files(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            args = request(output=str(root / "batch.png"), num_images=2, hires_save_base=True)
            base_json = root / "batch-0002-base.json"
            base_json.write_text("{}")
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                generation_output.resolve_output_paths(args)
            args.overwrite = True
            self.assertEqual([path.name for path in generation_output.resolve_output_paths(args)],
                             ["batch-0001.png", "batch-0002.png"])
            base_json.unlink()
            base_json.mkdir()
            with self.assertRaisesRegex(SystemExit, "directories"):
                generation_output.resolve_output_paths(args)

    def test_base_artifacts_do_not_reserve_names_when_base_saving_is_disabled(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            root = Path(temporary)
            (root / "image-base.png").write_bytes(b"older-base")
            args = request(output=str(root / "image.png"))
            self.assertEqual(generation_output.resolve_output_paths(args), [root / "image.png"])

    def test_main_publishes_refined_images_and_optional_base_metadata(self):
        cases = (
            (presets.SD15_PRESET, False, {}, "", ""),
            (presets.SD15_PRESET, True, {}, "", ""),
            (presets.FLUX1_SCHNELL_PRESET, True,
             {"true_cfg_scale": 1.0, "hires_true_cfg_scale": 2.0, "negative_prompt": "blurry"},
             "blurry", None),
            (presets.FLUX1_SCHNELL_PRESET, True,
             {"true_cfg_scale": 2.0, "hires_true_cfg_scale": 1.0, "negative_prompt": "blurry"},
             None, "blurry"),
        )
        for preset, save_base, options, final_negative, base_negative in cases:
            with self.subTest(preset=preset.name, save_base=save_base, options=options), \
                    tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
                root = Path(temporary)
                args = request(preset, output=str(root / "result.png"), hires_save_base=save_base,
                               cache_dir=str(root / "cache"), xet_cache_dir=str(root / "xet"),
                               seed=7, hires_seed=13, **options)
                events = []
                base = Pipeline(events, width=32, height=32)
                refined = Pipeline(events, width=64, height=64)
                torch = Torch(events)
                optimization = {"weight_storage": "cpu", "offload_policy": "none"}
                final_image = Image((64, 64), label="refined-final")
                observed = []

                def refine(pipeline, preset, values, images, *arguments, **keywords):
                    self.assertIs(pipeline, base)
                    self.assertEqual([image.size for image in images], [(32, 32)])
                    observed.extend(images)
                    second = hires_request(preset, values)
                    return hires.HiresResult(
                        refined, SimpleNamespace(images=[final_image]), second,
                        {"enabled": True, "refinement": {"clip_skip_layout_compatibility": False}},
                        optimization, {"class": "Scheduler"}, None, [13])

                with ExitStack() as stack:
                    replacements = {
                        "resolve_arguments": Mock(return_value=(preset, args)),
                        "package_versions": Mock(return_value={}),
                        "load_dependencies": Mock(return_value=(torch, {preset.pipeline_class: object()})),
                        "select_device": Mock(return_value="cpu"),
                        "configure_tensor_cores": Mock(return_value={}),
                        "accelerator_preflight": Mock(return_value={"runtime": "fixture", "backend": "cpu"}),
                        "load_generation_pipeline": Mock(return_value=(base, {
                            "component_sources": {"vae": "base_model"}, "vae_override": None})),
                        "prepare_pipeline_with_adapters": Mock(return_value=(base, optimization, None, None)),
                        "run_hires_fix": Mock(side_effect=refine),
                    }
                    for name, replacement in replacements.items():
                        stack.enter_context(patch.object(generate, name, replacement))
                    stack.enter_context(patch.object(sys, "argv", ["generate.py"]))
                    stack.enter_context(patch.dict("os.environ"))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    self.assertEqual(generate.main(), 0)
                self.assertEqual(len(observed), 1)
                self.assertEqual((root / "result.png").read_bytes(), final_image.tobytes())
                metadata = json.loads((root / "result.json").read_text())
                self.assertEqual(metadata["fixture"]["seed"], 13)
                self.assertEqual(metadata["fixture"]["negative_prompt"], final_negative)
                self.assertEqual((metadata["fixture"]["width"], metadata["fixture"]["height"]), (64, 64))
                self.assertEqual(metadata["hires_fix"]["base"]["seeds"], [7])
                self.assertEqual(metadata["hires_fix"]["base"]["executed_steps"], 1)
                self.assertEqual((root / "result-base.png").exists(), save_base)
                self.assertEqual((root / "result-base.json").exists(), save_base)
                if save_base:
                    base_metadata = json.loads((root / "result-base.json").read_text())
                    self.assertEqual(base_metadata["artifact_role"], "hires_base")
                    self.assertEqual(base_metadata["fixture"]["seed"], 7)
                    self.assertEqual(base_metadata["fixture"]["negative_prompt"], base_negative)
                    self.assertEqual(base_metadata["output"]["size"], [32, 32])
                    self.assertEqual(base_metadata["final_output"]["size"], [64, 64])


if __name__ == "__main__":
    unittest.main()
