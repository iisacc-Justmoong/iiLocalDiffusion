#!/usr/bin/env python3
"""Dependency-free contracts for the generic installed Diffusers runner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate_any as generic


class FakePipeline:
    __module__ = "diffusers.pipelines.test_builtin"

    def __call__(self, prompt=None, image=None, width=32, num_inference_steps=1,
                 guidance_scale=1.0, generator=None, output_type="pil", **kwargs):
        pass


class FakeImage:
    is_animated = False
    mode = "RGB"
    size = (32, 32)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load(self):
        pass

    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, path, **kwargs):
        Path(path).write_bytes(b"fake-png-contract-only")


class GenericDiffusersTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="generic-tests-")
        self.directory = Path(self.temporary.name)
        self.model = self.directory / "model"
        self.model.mkdir()
        self.index = {"_class_name": "FakePipeline", "unet": ["diffusers", "UNet2DModel"]}
        (self.model / "model_index.json").write_text(json.dumps(self.index))
        self.fake_images = SimpleNamespace(Image=FakeImage, open=Mock(return_value=FakeImage()))

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, *tokens):
        return generic.resolve_arguments(generic.build_parser().parse_args([
            "--model", str(self.model), "--output-dir", str(self.directory / "output"), *tokens]))

    def fake_diffusers(self):
        class Base:
            pass
        class Builtin(Base, FakePipeline):
            __module__ = "diffusers.pipelines.test_builtin"
        Builtin.__name__ = "FakePipeline"
        Builtin.from_pretrained = Mock(side_effect=lambda *args, **kwargs: Builtin())
        Builtin.from_single_file = Mock(side_effect=lambda *args, **kwargs: Builtin())
        return SimpleNamespace(DiffusionPipeline=Base, FakePipeline=Builtin, pipelines=SimpleNamespace())

    def test_omission_preserves_pipeline_defaults_without_injecting_prompt_or_seed(self):
        args = self.args()
        self.assertEqual(args.inputs, {})
        self.assertIsNone(args.seed)
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.offload, "none")

    def test_explicit_cli_inputs_override_json_without_discarding_task_values(self):
        args = self.args("--pipeline-inputs", '{"prompt":"old","num_frames":17,"guidance_scale":3}',
                         "--prompt", "", "--guidance-scale", "0", "--steps", "1")
        self.assertEqual(args.inputs, {"prompt": "", "num_frames": 17, "guidance_scale": 0.0,
                                      "num_inference_steps": 1})

    def test_inputs_json_file(self):
        path = self.directory / "inputs.json"
        path.write_text('{"prompt_2":"secondary","strength":0.2}')
        self.assertEqual(self.args("--pipeline-inputs", "@" + str(path)).inputs,
                         {"prompt_2": "secondary", "strength": 0.2})

    def test_rejects_non_object_duplicates_nonfinite_and_latent_requests(self):
        for value in ('[]', '{"x":NaN}', '{"nested":{"x":1e999}}', '{"return_dict":null}', '{"x":1,"x":2}', '{"return_dict":false}',
                      '{"output_type":"latent"}', '{"generator":1}', '{"_private":true}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.args("--pipeline-inputs", value)

    def test_remote_models_require_an_immutable_revision(self):
        for revision in (None, "main", "abcdef", "A" * 40):
            tokens = ["--model", "org/repository"] + (["--revision", revision] if revision else [])
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "immutable"):
                generic.resolve_arguments(generic.build_parser().parse_args(tokens))
        args = generic.resolve_arguments(generic.build_parser().parse_args([
            "--model", "org/repository", "--revision", "a" * 40]))
        self.assertEqual(args.source_kind, "hub")

    def test_missing_local_files_are_not_reinterpreted_as_hub_repositories(self):
        for model in ("/missing/model", "./missing", "weights/missing.safetensors", "weights/model.gguf"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                generic.resolve_arguments(generic.build_parser().parse_args([
                    "--model", model, "--revision", "a" * 40]))

    def test_local_revision_and_single_file_config_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "remote"):
            self.args("--revision", "a" * 40)
        with self.assertRaisesRegex(ValueError, "only valid"):
            self.args("--model-config", str(self.model))

    def test_single_file_requires_safetensors_and_complete_local_config(self):
        for name in ("model.ckpt", "model.gguf", "model.safetensor"):
            path = self.directory / name
            path.write_bytes(b"weights")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "safetensors"):
                self.args("--model", str(path))
        path = self.directory / "model.safetensors"
        path.write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, "model-config"):
            self.args("--model", str(path))
        args = self.args("--model", str(path), "--model-config", str(self.model))
        self.assertEqual(args.source_kind, "single-file")

    def test_invalid_seeds_sizes_guidance_and_cpu_offload_fail_preflight(self):
        for tokens in (("--seed", "-1"), ("--seed", str(2**63)), ("--width", "0"),
                       ("--height", "-2"), ("--steps", "0"), ("--guidance-scale", "nan"),
                       ("--audio-sample-rate", "0"), ("--device", "cpu", "--offload", "model")):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.args(*tokens)

    def test_print_config_requires_no_third_party_python_packages(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "reference/diffusers/generate_any.py"),
                                 "--model", str(self.model), "--print-config"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["pipeline_inputs"], {})

    def test_model_index_custom_code_and_paths_are_rejected(self):
        for value in ({"_class_name": ["evil", "Evil"]}, {"_class_name": "mod.Evil"},
                      {"_class_name": "Good", "unet": ["os.path", "Any"]},
                      {"_class_name": "Good", "../unet": ["diffusers", "Any"]},
                      {"_class_name": "Good", "unet": ["diffusers"]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generic.validate_model_index(value)

    def test_arbitrary_installed_modules_are_not_allowed_as_components(self):
        index = generic.validate_model_index({"_class_name": "FakePipeline", "unet": ["os", "system"]})
        with self.assertRaisesRegex(ValueError, "non-built-in"):
            generic.validate_components(self.fake_diffusers(), index)

    def test_custom_local_component_code_is_not_executed(self):
        directory = self.model / "unet"
        directory.mkdir()
        (directory / "diffusers.py").write_text("raise RuntimeError('must not execute')")
        with self.assertRaisesRegex(ValueError, "Custom local component"):
            generic.validate_components(self.fake_diffusers(), self.index, self.model)

    def test_only_public_installed_diffusers_pipeline_classes_can_be_selected(self):
        diffusers = self.fake_diffusers()
        self.assertEqual(generic.builtin_pipeline(diffusers, "FakePipeline"), diffusers.FakePipeline)
        diffusers.Evil = FakePipeline
        diffusers.NotPipeline = str
        for name in ("Evil", "NotPipeline", "Missing", "DiffusionPipeline"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                generic.builtin_pipeline(diffusers, name)

    def test_civitai_labels_reject_cross_architecture_models(self):
        diffusers = self.fake_diffusers()
        cases = (("SD 1.5", "StableDiffusionPipeline"), ("Illustrious", "StableDiffusionXLPipeline"),
                 ("SD 3.5", "StableDiffusion3Pipeline"), ("Flux.1 Krea", "FluxPipeline"),
                 ("Flux.2 D", "Flux2Pipeline"), ("Qwen", "QwenImagePipeline"))
        classes = {}
        for label, name in cases:
            classes[label] = type(name, (diffusers.DiffusionPipeline,),
                                  {"__module__": "diffusers.pipelines.test_builtin"})
            setattr(diffusers, name, classes[label])
        for label, _ in cases:
            for other, _ in cases:
                with self.subTest(label=label, actual=other):
                    if label == other:
                        result = generic.validate_base_model(label, classes[other], diffusers)
                        self.assertEqual(result["status"], "pipeline-architecture-verified")
                    else:
                        with self.assertRaisesRegex(ValueError, "requires architecture"):
                            generic.validate_base_model(label, classes[other], diffusers)

    def test_task_siblings_remain_compatible_across_diffusers_namespaces(self):
        diffusers = self.fake_diffusers()
        for label, expected, sibling in (
                ("NoobAI", "StableDiffusionXLPipeline", "StableDiffusionXLInpaintPipeline"),
                ("SD 3", "StableDiffusion3Pipeline", "StableDiffusion3ControlNetPipeline"),
                ("Flux.1 D", "FluxPipeline", "FluxKontextPipeline"),
                ("Qwen", "QwenImagePipeline", "QwenImageEditPipeline")):
            setattr(diffusers, expected, type(expected, (diffusers.DiffusionPipeline,),
                    {"__module__": "diffusers.pipelines.base_namespace"}))
            actual = type(sibling, (diffusers.DiffusionPipeline,),
                          {"__module__": "diffusers.pipelines.task_namespace"})
            with self.subTest(label=label):
                result = generic.validate_base_model(label, actual, diffusers)
                self.assertEqual(result["status"], "pipeline-architecture-verified")

    def test_kolors_task_siblings_share_the_builtin_kolors_namespace(self):
        diffusers = self.fake_diffusers()
        diffusers.KolorsPipeline = type("KolorsPipeline", (diffusers.DiffusionPipeline,),
                                        {"__module__": "diffusers.pipelines.kolors.pipeline_kolors"})
        image_pipeline = type("KolorsImg2ImgPipeline", (diffusers.DiffusionPipeline,),
                              {"__module__": "diffusers.pipelines.kolors.pipeline_kolors_img2img"})
        report = generic.validate_base_model("Kolors", image_pipeline, diffusers)
        self.assertEqual(report["status"], "pipeline-architecture-verified")
        self.assertEqual(report["actual_architecture"], "diffusers:kolors")

    def test_single_file_cannot_fall_back_to_directory_loader_when_class_has_no_loader(self):
        path = self.directory / "download.safetensors"
        path.write_bytes(b"fixture")
        args = self.args("--model", str(path), "--model-config", str(self.model))
        diffusers = self.fake_diffusers()
        delattr(diffusers.FakePipeline, "from_single_file")
        with self.assertRaisesRegex(ValueError, "no built-in single-file loader"):
            generic.load_pipeline(args, diffusers, "float32")
        diffusers.FakePipeline.from_pretrained.assert_not_called()

    def test_workflow_family_without_known_class_is_explicitly_declared_only(self):
        result = generic.validate_base_model("Anima", FakePipeline, self.fake_diffusers())
        self.assertEqual(result["status"], "declared-only")
        with self.assertRaisesRegex(ValueError, "Unknown Civitai"):
            generic.validate_base_model("invented unknown label", FakePipeline, self.fake_diffusers())

    def test_unknown_kwargs_are_rejected_even_when_pipeline_accepts_arbitrary_kwargs(self):
        with self.assertRaisesRegex(ValueError, "typo"):
            generic.validate_call_arguments(FakePipeline().__call__, {"typo": 1}, seed=None)
        generic.validate_call_arguments(FakePipeline().__call__, {"prompt": "hi"}, seed=0)

    def test_required_inputs_and_unsupported_seed_are_explicit(self):
        def requires_image(image, *, strength=0.8):
            pass
        with self.assertRaisesRegex(ValueError, "requires input.*image"):
            generic.validate_call_arguments(requires_image, {}, seed=None)
        with self.assertRaisesRegex(ValueError, "generator"):
            generic.validate_call_arguments(requires_image, {"image": "file"}, seed=1)

    def test_directory_load_is_local_safe_and_records_actual_pipeline(self):
        diffusers = self.fake_diffusers()
        pipeline, report = generic.load_pipeline(self.args(), diffusers, "float32")
        self.assertIsInstance(pipeline, diffusers.FakePipeline)
        kwargs = diffusers.FakePipeline.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["local_files_only"], True)
        self.assertEqual(kwargs["use_safetensors"], True)
        self.assertEqual(kwargs["trust_remote_code"], False)
        self.assertEqual(report["actual_class"], "FakePipeline")
        self.assertTrue(report["identity"]["files"][0]["sha256"])

    def test_invalid_kwargs_fail_before_weight_loading(self):
        diffusers = self.fake_diffusers()
        with self.assertRaisesRegex(ValueError, "misspelled"):
            generic.load_pipeline(self.args("--pipeline-inputs", '{"misspelled":1}'), diffusers, "dtype")
        diffusers.FakePipeline.from_pretrained.assert_not_called()

    def test_single_file_passes_explicit_local_config_and_refuses_pickle_extras(self):
        diffusers = self.fake_diffusers()
        path = self.directory / "model.safetensors"
        path.write_bytes(b"safe fixture")
        args = self.args("--model", str(path), "--model-config", str(self.model))
        _, report = generic.load_pipeline(args, diffusers, "dtype")
        self.assertEqual(diffusers.FakePipeline.from_single_file.call_args.kwargs["config"], str(self.model))
        self.assertEqual(report["identity"]["single_file"]["size_bytes"], len(b"safe fixture"))
        (self.model / "pytorch_model.bin").write_bytes(b"unsafe fixture")
        with self.assertRaisesRegex(ValueError, "pickle"):
            generic.load_pipeline(args, diffusers, "dtype")

    def test_model_identity_detects_replacement_with_equal_byte_size(self):
        path = self.model / "weights.safetensors"
        path.write_bytes(b"old")
        identity = generic.model_identity(self.model)
        path.write_bytes(b"new")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            generic.verify_identity(identity)

    def test_image_paths_decode_only_in_controlled_input_fields_and_preserve_batch_order(self):
        path = self.directory / "image.png"
        path.write_bytes(b"image fixture")
        inputs, files = generic.prepare_inputs({"image": [{"image_path": str(path), "mode": "L"}]}, self.fake_images)
        self.assertIsInstance(inputs["image"][0], FakeImage)
        self.assertEqual(files[0]["mode"], "L")
        self.assertEqual(files[0]["input"], "image")
        for value in ({"prompt": {"image_path": str(path)}},
                      {"image": {"image_path": str(path), "other": 1}},
                      {"image": {"image_path": str(path), "mode": "CMYK"}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generic.prepare_inputs(value, self.fake_images)

    def test_animated_images_require_explicit_frame_list(self):
        path = self.directory / "image.gif"
        path.write_bytes(b"fixture")
        self.fake_images.open.return_value.is_animated = True
        with self.assertRaisesRegex(ValueError, "Animated"):
            generic.prepare_inputs({"video": {"image_path": str(path)}}, self.fake_images)

    def test_image_and_video_batches_export_all_frames_with_provenance(self):
        args = self.args()
        outputs = generic.save_outputs(SimpleNamespace(images=[FakeImage(), FakeImage()],
                                        frames=[[FakeImage(), FakeImage()], [FakeImage()]]),
                                       None, args, self.fake_images, None)
        self.assertEqual(len(outputs), 5)
        self.assertEqual(sum(item["kind"] == "video-frame" for item in outputs), 3)
        self.assertTrue(all(Path(item["path"]).is_file() and item["sha256"] for item in outputs))

    def test_video_layout_preserves_short_frame_counts_without_channel_guessing(self):
        class Array:
            def __init__(self, shape):
                self.shape, self.ndim = shape, len(shape)
            def transpose(self, *axes):
                return Array(tuple(self.shape[axis] for axis in axes))
            def __len__(self):
                return self.shape[0]
            def __iter__(self):
                for _ in range(self.shape[0]):
                    yield [FakeImage() for _ in range(self.shape[1])]
        class Tensor(Array):
            def detach(self):
                return self
            def cpu(self):
                return self
            def float(self):
                return self
            def is_floating_point(self):
                return True
            def numpy(self):
                return Array(self.shape)
        np = SimpleNamespace(asarray=lambda value: value)
        for value, layout in ((Tensor((1, 4, 3, 16, 16)), "auto"),
                              (Array((1, 4, 16, 16, 3)), "auto"),
                              (Tensor((1, 3, 4, 16, 16)), "bcfhw")):
            with self.subTest(layout=layout, shape=value.shape):
                outputs = generic.save_outputs({"frames": value}, None,
                    self.args("--overwrite", "--video-layout", layout), self.fake_images, np)
                self.assertEqual(len(outputs), 4)
        with self.assertRaisesRegex(ValueError, "invalid channel axis"):
            generic.save_outputs({"frames": Array((1, 4, 3, 16, 16))}, None,
                                 self.args(), self.fake_images, np)

    def test_unknown_outputs_do_not_claim_success(self):
        with self.assertRaisesRegex(ValueError, "no supported media"):
            generic.save_outputs(SimpleNamespace(meshes=["mesh"]), None, self.args(), self.fake_images, None)

    def test_empty_declared_media_is_not_accepted_as_a_complete_batch(self):
        for result in ({"images": []}, {"frames": []}, {"frames": [[]]},
                       {"images": [FakeImage()], "frames": [[]]}):
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "empty"):
                generic.save_outputs(result, None, self.args("--overwrite"), self.fake_images, None)

    def test_overwrite_failure_archives_old_report_before_changing_artifacts(self):
        class FailingImage(FakeImage):
            def save(self, path, **kwargs):
                raise RuntimeError("second image encoding failed")
        args = self.args("--overwrite")
        args.output_dir.mkdir()
        prior = '{"old":"report remains preserved"}'
        (args.output_dir / "generation.json").write_text(prior)
        (args.output_dir / "image-0001.png").write_bytes(b"old pixels")
        (args.output_dir / "unrelated.txt").write_text("preserve")
        with self.assertRaisesRegex(RuntimeError, "second image"):
            generic.publish_generation({"images": [FakeImage(), FailingImage()]}, None, args,
                                       self.fake_images, None, {"request": "new"})
        self.assertFalse((args.output_dir / "generation.json").exists())
        archives = list(args.output_dir.glob("generation.previous-*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_text(), prior)
        self.assertEqual((args.output_dir / "image-0001.png").read_bytes(), b"fake-png-contract-only")
        self.assertEqual((args.output_dir / "unrelated.txt").read_text(), "preserve")

    def test_completed_overwrite_publishes_new_report_with_actual_safety_flags(self):
        args = self.args("--overwrite")
        args.output_dir.mkdir()
        (args.output_dir / "generation.json").write_text('{"old":true}')
        old_archive = args.output_dir / "generation.previous-existing.json"
        old_archive.write_text('{"older":true}')
        path, outputs = generic.publish_generation(
            {"images": [FakeImage(), FakeImage()], "nsfw_content_detected": [False, True]},
            SimpleNamespace(safety_checker=object()), args, self.fake_images, None, {"request": "new"})
        report = json.loads(path.read_text())
        self.assertEqual(report["request"], "new")
        self.assertEqual(len(outputs), 2)
        self.assertEqual(report["safety"], {"checker_present": True,
                         "pipeline_report": {"nsfw_content_detected": [False, True]}})
        self.assertTrue(Path(report["previous_report"]).is_file())
        self.assertEqual(old_archive.read_text(), '{"older":true}')

    def test_misaligned_safety_flags_leave_no_success_report(self):
        args = self.args()
        with self.assertRaisesRegex(ValueError, "flags do not match"):
            generic.publish_generation({"images": [FakeImage()], "nsfw_content_detected": [True, False]},
                                       None, args, self.fake_images, None, {})
        self.assertFalse((args.output_dir / "generation.json").exists())

    def test_absent_safety_checker_is_not_reported_as_successful_screening(self):
        report = generic.safety_metadata({}, None, [{"kind": "image"}])
        self.assertEqual(report, {"checker_present": False, "pipeline_report": {}})

    def test_atomic_output_collision_does_not_overwrite_existing_content(self):
        path = self.directory / "image.png"
        path.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            generic.atomic_write(path, lambda temporary: temporary.write_bytes(b"new"), False)
        self.assertEqual(path.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])
        generic.atomic_write(path, lambda temporary: temporary.write_bytes(b"new"), True)
        self.assertEqual(path.read_bytes(), b"new")

    def test_failed_writer_cleans_its_temporary_file(self):
        with self.assertRaises(RuntimeError):
            generic.atomic_write(self.directory / "failed.png", Mock(side_effect=RuntimeError("encoding failed")), False)
        self.assertEqual(list(self.directory.glob(".*.tmp")), [])

    def test_audio_rate_is_explicit_or_read_from_the_actual_component(self):
        pipeline = SimpleNamespace(vocoder=SimpleNamespace(config=SimpleNamespace(sampling_rate=16000)))
        self.assertEqual(generic.infer_sample_rate(pipeline, None), 16000)
        self.assertEqual(generic.infer_sample_rate(pipeline, 48000), 48000)
        with self.assertRaisesRegex(ValueError, "audio-sample-rate"):
            generic.infer_sample_rate(SimpleNamespace(), None)

    def test_hardware_policy_checks_offloaded_execution_device(self):
        pipeline = Mock(_execution_device=SimpleNamespace(type="mps"))
        generic.prepare_execution(pipeline, "mps", "model")
        pipeline.enable_model_cpu_offload.assert_called_once_with(device="mps")
        pipeline.to.assert_not_called()
        pipeline._execution_device.type = "cpu"
        with self.assertRaisesRegex(RuntimeError, "selected hardware"):
            generic.prepare_execution(pipeline, "mps", "sequential")


if __name__ == "__main__":
    unittest.main()
