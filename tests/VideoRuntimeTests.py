#!/usr/bin/env python3
"""Video tensor, keyframe, model identity and transactional output contracts."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from local_model_fixture import local_parser
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from animation_video import AnimationOutput, output_targets, SCHEMAS
from video_options import build_parser, resolve_options
from video_runtime import model_contract, verify_model_files


class VideoModelTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="video-runtime-", dir=ROOT / "build")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_image_models_and_custom_code_are_not_temporal_models(self):
        path = self.directory / "model_index.json"
        path.write_text(json.dumps({"_class_name": "StableDiffusionPipeline"}))
        with self.assertRaisesRegex(ValueError, "temporal"):
            model_contract(self.directory)
        path.write_text(json.dumps({"_class_name": "LTXPipeline", "scheduler": ["custom", "Scheduler"]}))
        with self.assertRaisesRegex(ValueError, "component"):
            model_contract(self.directory)

    def test_modified_model_files_cannot_be_published_with_old_identity(self):
        path = self.directory / "weights.safetensors"
        path.write_bytes(b"weights")
        stat = path.stat()
        stamps = {path: (stat.st_size, stat.st_mtime_ns)}
        verify_model_files(stamps)
        path.write_bytes(b"different weights")
        with self.assertRaises(RuntimeError):
            verify_model_files(stamps)

    def test_camera_caption_cannot_be_silently_truncated(self):
        from video_runtime import validate_captions
        pipeline = SimpleNamespace(tokenizer=lambda *a, **k: {"input_ids": list(range(70))})
        args = resolve_options(local_parser(build_parser).parse_args(["--max-sequence-length", "64"]))
        with self.assertRaisesRegex(ValueError, "Silent truncation"):
            validate_captions(pipeline, args)

    def test_sentencepiece_dependency_failure_identifies_video_requirements(self):
        from video_runtime import load_tokenizer
        tokenizer = self.directory / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "spiece.model").write_bytes(b"sentencepiece")
        with patch.dict(sys.modules, {"google.protobuf": None}):
            with self.assertRaisesRegex(ImportError, "requirements-video.txt"):
                load_tokenizer(self.directory, {"tokenizer": ["transformers", "T5Tokenizer"]})

    def test_temporal_bundle_rolls_back_on_publication_failure(self):
        args = resolve_options(local_parser(build_parser).parse_args(["--output", str(self.directory / "movie.mp4")]))
        def stage(output, content):
            output.video.write_bytes(content)
            (output.frames / "frame-000000.png").write_bytes(content)
        with AnimationOutput(args) as output:
            stage(output, b"old")
            output.commit({"status": "old"})
        args.overwrite = True
        with self.assertRaises(OSError), patch("animation_video.publish_file", side_effect=OSError("disk full")):
            with AnimationOutput(args) as output:
                stage(output, b"new")
                output.commit({"status": "new"})
        video, report, frames = output_targets(args.output)
        self.assertEqual(video.read_bytes(), b"old")
        self.assertEqual(json.loads(report.read_text())["status"], "old")
        self.assertEqual(json.loads((frames / "manifest.json").read_text())["schema"], SCHEMAS["Video"])
        self.assertFalse(list(self.directory.glob(".*.lock")))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "PyTorch and Diffusers are required")
class VideoTensorTests(unittest.TestCase):
    def setUp(self):
        VideoModelTests.setUp(self)
        import torch
        import numpy as np
        from PIL import Image
        self.torch, self.np, self.Image = torch, np, Image

    def test_tokenizer_is_checked_before_large_model_files_are_read(self):
        from video_runtime import load_pipeline
        args = SimpleNamespace(model=str(self.directory))
        with patch("video_runtime.model_contract", return_value={}), \
                patch("video_runtime.load_tokenizer", side_effect=ValueError("invalid tokenizer")), \
                patch("video_runtime.file_sha256") as hash_file:
            with self.assertRaisesRegex(ValueError, "invalid tokenizer"):
                load_pipeline(args, self.torch, self.torch.float32)
            hash_file.assert_not_called()

    def test_finite_audit_rejects_corrupt_latents(self):
        from video_runtime import VideoDenoisingAudit
        audit = VideoDenoisingAudit(self.torch, "cpu")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            audit(None, 0, 1, {"latents": self.torch.tensor([float("nan")])})
        self.assertEqual(audit.steps, [])

    def test_keyframe_orientation_size_and_hash_are_preserved_in_provenance(self):
        from video_runtime import load_keyframes
        image = self.directory / "image.png"
        self.Image.new("RGB", (64, 32), "red").save(image)
        args = resolve_options(local_parser(build_parser).parse_args(["--first-frame", str(image), "--width", "32", "--height", "32"]))
        loaded, metadata = load_keyframes(args, self.Image)
        self.assertEqual(loaded[str(image)].size, (32, 32))
        self.assertEqual(metadata[str(image)]["original_size"], [64, 32])
        self.assertEqual(len(metadata[str(image)]["sha256"]), 64)

    def test_temporal_sampling_uses_source_plan_and_preserves_shot_continuity(self):
        from video_runtime import render_shots
        story = self.directory / "shots.json"
        story.write_text(json.dumps({"shots": [{"frames": 8}, {"frames": 10, "continue_previous": True}]}))
        args = resolve_options(local_parser(build_parser).parse_args(["--storyboard", str(story), "--width", "32", "--height", "32", "--steps", "2"]))
        calls = []
        def pipeline(**kwargs):
            calls.append(kwargs)
            for step in range(2):
                kwargs["callback_on_step_end"](None, step, 1 - step, {"latents": self.torch.zeros((1, 2, 4))})
            frames = self.np.zeros((1, kwargs["num_frames"], 32, 32, 3), dtype=self.np.float32)
            frames[:, :, :, :, 0] = self.np.arange(kwargs["num_frames"])[None, :, None, None] / 20
            return SimpleNamespace(frames=frames)
        output = SimpleNamespace(frames=self.directory)
        records, frames = render_shots(pipeline, args, self.torch, "cpu", [], {}, output, self.Image)
        self.assertEqual([call["num_frames"] for call in calls], [9, 9])
        self.assertEqual(len(frames), 11)
        self.assertEqual([record["start_frame"] for record in records], [0, 8])
        self.assertIsNone(calls[0]["conditions"])
        self.assertEqual(calls[1]["conditions"][0].image.getpixel((0, 0))[0], round(4 / 20 * 255))
        self.assertEqual([call["frame_rate"] for call in calls], [24 * 4 / 7, 24 * 5 / 9])
        self.assertEqual([f["index"] for f in frames], [0, 2, 4, 6, 7, 8, 10, 12, 14, 16, 17])

    def test_bad_decoded_values_do_not_become_black_frames(self):
        from video_runtime import render_shots
        args = resolve_options(local_parser(build_parser).parse_args(["--frames", "9", "--width", "32", "--height", "32"]))
        frames = self.np.full((1, 9, 32, 32, 3), float("nan"), dtype=self.np.float32)
        with self.assertRaisesRegex(RuntimeError, "finite RGB"):
            render_shots(lambda **kwargs: SimpleNamespace(frames=frames), args, self.torch,
                         "cpu", [], {}, SimpleNamespace(frames=self.directory), self.Image)
        self.assertFalse(list(self.directory.glob("frame-*.png")))


if __name__ == "__main__":
    unittest.main()
