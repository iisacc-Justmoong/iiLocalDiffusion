#!/usr/bin/env python3
"""Feedback, adapter, sampling and publication contracts without model weights."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
import deforum_runtime as runtime
from deforum_video import AnimationOutput, SCHEMA, output_targets
from HiresRuntimeTests import Image, Pipeline, Torch


class Frame(Image):
    def copy(self):
        return Frame(self.size, label=self.label)


class FramePipeline(Pipeline):
    def __call__(self, **call):
        result = super().__call__(**call)
        if self.output is None:
            initial = call.get("image")
            label = "text" if initial is None else initial.label
            result.images = [Frame((self.width, self.height), label=f"{label}:diffused-{len(self.calls)}")]
        return result


class DeforumRuntimeTests(unittest.TestCase):
    def run_frames(self, *, preset="sd15", initial=None, strength="0:(0.5)", controlnet=False,
                   pipeline=None, activation=None, **values):
        selected, args = generate.resolve_request({"preset": preset, "animation_mode": "2D",
                                                   "max_frames": 3, "width": 32, "height": 32,
                                                   "steps": 4, "device": "cpu",
                                                   "strength_schedule": strength, **values})
        if controlnet:
            args.controlnet_selection = object()
            args.controlnet_image = Frame(label="control")
        pipeline = pipeline or FramePipeline([], width=32, height=32)
        self.conversion = Mock()

        def from_pipe(source, **overrides):
            self.assertTrue(source.hooks_removed)
            self.assertTrue(source.on_cpu)
            self.conversion(source)
            source.scheduler = overrides["scheduler"]
            return source

        classes = {name: SimpleNamespace(from_pipe=from_pipe) for name in runtime.PIPELINES.values()}
        self.transforms, self.saved, self.schedulers = [], [], []

        def transform(source, reference, request, environment, *, warp):
            self.transforms.append((source, reference, request.frame_index, warp))
            return Frame(source.size, label=source.label + ":warped")

        def write(index, image):
            self.saved.append(image)
            self.schedulers.append(pipeline.scheduler)
            return {"path": f"frame-{index:06d}.png"}

        with redirect_stdout(io.StringIO()):
            frames = runtime.render_deforum_frames(
                pipeline, selected, args, Torch(), "cpu", "float32", False, activation,
                build_call=generate.build_pipeline_call_arguments,
                prepare_execution=lambda pipe, *a, **kw: (pipe, {}),
                environment={"initial_image": initial}, write_frame=write,
                pipeline_classes=classes, transform=transform)
        return pipeline, frames

    def test_each_frame_uses_preceding_output_and_a_fresh_scheduler(self):
        pipeline, frames = self.run_frames(animation_prompts={"0": "city", "2": "forest"}, seed_behavior="iter")
        self.assertNotIn("image", pipeline.calls[0])
        self.assertEqual(pipeline.calls[1]["image"].label, self.saved[0].label + ":warped")
        self.assertEqual(pipeline.calls[2]["image"].label, self.saved[1].label + ":warped")
        self.assertEqual([call["prompt"] for call in pipeline.calls], ["city", "city", "forest"])
        self.assertEqual([frame["seed"] for frame in frames], [42, 43, 44])
        self.assertEqual([frame["sampling"]["executed_steps"] for frame in frames], [1, 1, 1])
        self.assertEqual(len({id(scheduler) for scheduler in self.schedulers}), 3)
        self.assertEqual(self.conversion.call_count, 1)
        self.assertEqual(frames[2]["source"]["pixel_sha256"], frames[1]["output"]["pixel_sha256"])

    def test_initial_image_enters_img2img_without_first_frame_camera_motion(self):
        pipeline, frames = self.run_frames(initial=Frame(label="initial"))
        self.assertEqual(frames[0]["method"], "image-to-image")
        self.assertFalse(frames[0]["camera_applied"])
        self.assertTrue(frames[1]["camera_applied"])
        self.assertFalse(self.transforms[0][3])
        self.assertTrue(self.transforms[1][3])
        self.assertEqual(self.conversion.call_count, 1)

    def test_zero_strength_is_explicit_warp_only_and_does_not_denoise(self):
        pipeline, frames = self.run_frames(strength="0:(0)")
        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual([frame["method"] for frame in frames], ["text-to-image", "warp-only", "warp-only"])
        self.assertEqual(frames[1]["sampling"]["executed_steps"], 0)
        self.assertFalse(self.conversion.called)

    def test_nonfinite_or_missing_sampling_cannot_publish_a_frame(self):
        for field in ("finite", "invoke_callback"):
            pipeline = FramePipeline([], width=32, height=32)
            setattr(pipeline, field, False)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.run_frames(pipeline=pipeline)
            self.assertEqual(self.saved, [])

    def test_adapter_loss_is_detected_before_second_frame(self):
        pipeline = FramePipeline([], width=32, height=32, active=())
        activation = SimpleNamespace(registered_components=["unet"], active_adapters=["iild_lora"])
        with self.assertRaisesRegex(RuntimeError, "lost active LoRA"):
            self.run_frames(pipeline=pipeline, activation=activation)
        self.assertEqual(len(self.saved), 1)

    def test_sdxl_and_flux_keep_secondary_prompts(self):
        for preset in ("sdxl-base", "flux1-schnell"):
            with self.subTest(preset=preset):
                pipeline, _ = self.run_frames(preset=preset, animation_prompts={"0": "a", "2": "b"})
                self.assertEqual(pipeline.calls[2]["prompt_2"], "b")
                self.assertEqual(pipeline.calls[2]["strength"], 0.5)

    def test_control_image_is_distinct_from_feedback_image(self):
        for preset in ("sd15", "sdxl-base"):
            pipeline, _ = self.run_frames(preset=preset, controlnet=True)
            self.assertEqual(pipeline.calls[1]["control_image"].label, "control")
            self.assertNotEqual(pipeline.calls[1]["image"].label, "control")

    def test_flux_controlnet_uses_the_existing_noise_schedule_adapter(self):
        def refine(pipeline, request, images, call, *unused):
            self.assertEqual(len(images), 1)
            self.assertEqual(request.hires_denoising_strength, 0.5)
            self.assertNotIn("image", call)
            self.assertEqual(call["control_image"].label, "control")
            return pipeline(**call), {"conditioning": "img2img-latents-with-controlnet-and-true-cfg"}

        with patch("hires_flux_controlnet.refine_flux_controlnet", side_effect=refine) as adapter:
            _, frames = self.run_frames(preset="flux1-schnell", controlnet=True,
                                        true_cfg_scale=2, animation_negative_prompts={"0": "blur"})
        self.assertEqual(adapter.call_count, 2)
        self.assertEqual(frames[1]["compatibility"]["conditioning"],
                         "img2img-latents-with-controlnet-and-true-cfg")

    def test_cpu_prompt_encoding_is_refreshed_for_each_frame(self):
        conditioning = SimpleNamespace(metadata={"enabled": True},
                                       for_device=lambda *a: {"prompt_embeds": "embedded"})
        with patch.object(runtime, "encode_cpu_prompt", return_value=conditioning) as encode:
            self.run_frames(cpu_text_encoding=True, animation_prompts={"0": "a", "1": "b"})
        self.assertEqual([call.args[2].prompt for call in encode.call_args_list], ["a", "b", "b"])


class AnimationPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="deforum-publication-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = SimpleNamespace(output=self.directory / "movie.mp4", overwrite=False, max_frames=2)

    def stage(self, output):
        output.video.write_bytes(b"movie")
        (output.frames / "frame-000000.png").write_bytes(b"frame")

    def test_sampling_failure_leaves_no_final_files_or_lock(self):
        with self.assertRaisesRegex(RuntimeError, "sampling"):
            with AnimationOutput(self.args) as output:
                self.stage(output)
                raise RuntimeError("sampling")
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_completed_bundle_and_nonoverwrite_collision(self):
        with AnimationOutput(self.args) as output:
            self.stage(output)
            output.commit({"status": "complete"})
        video, report, frames = output_targets(self.args.output)
        self.assertEqual(video.read_bytes(), b"movie")
        self.assertEqual(json.loads(report.read_text())["status"], "complete")
        self.assertEqual(json.loads((frames / "manifest.json").read_text())["schema"], SCHEMA)
        with self.assertRaises(ValueError):
            with AnimationOutput(self.args):
                pass

    def test_overwrite_failure_restores_the_entire_previous_bundle(self):
        with AnimationOutput(self.args) as output:
            self.stage(output)
            output.commit({"status": "old"})
        self.args.overwrite = True
        with self.assertRaises(OSError):
            with AnimationOutput(self.args) as output:
                self.stage(output)
                with patch("animation_video.publish_file", side_effect=OSError("disk full")):
                    output.commit({"status": "new"})
        video, report, frames = output_targets(self.args.output)
        self.assertEqual(video.read_bytes(), b"movie")
        self.assertEqual(json.loads(report.read_text())["status"], "old")
        self.assertTrue((frames / "frame-000000.png").is_file())

    def test_unmanaged_directory_is_preserved_even_with_overwrite(self):
        self.args.overwrite = True
        _, _, frames = output_targets(self.args.output)
        frames.mkdir()
        (frames / "user.txt").write_text("preserve")
        with self.assertRaises(ValueError):
            with AnimationOutput(self.args):
                pass
        self.assertEqual((frames / "user.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
