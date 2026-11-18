#!/usr/bin/env python3
"""Cinematic video planning and front-door contracts, without model downloads."""
import resource_fixture

import json
import pickle
from pathlib import Path
import subprocess
import struct
import sys
import unittest
from local_model_fixture import local_parser, MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
from video_options import build_parser, resolve_options, configuration, CAMERA_MOTIONS


class VideoOptionsTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        temporary = resource_fixture.TemporaryDirectory(prefix="video-options-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def request(self, *tokens):
        return resolve_options(local_parser(build_parser).parse_args(list(tokens)))

    def document(self, name, data):
        path = self.directory / name
        path.write_text(json.dumps(data))
        return path

    def test_text_encoder_formats_allow_gguf_and_safetensors_but_not_onnx(self):
        for suffix in (".gguf", ".safetensors", ".safetensor"):
            with self.subTest(suffix=suffix):
                model = self.directory / ("encoder" + suffix)
                model.write_bytes(b"fixture")
                self.assertEqual(self.request("--text-encoder", str(model)).text_encoder, model.resolve())
        for suffix in (".onnx", ".ONNX"):
            model = self.directory / ("encoder" + suffix)
            model.write_bytes(b"fixture")
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.request("--text-encoder", str(model))

    def test_renamed_video_and_encoder_containers_follow_their_content(self):
        header = b'{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        safe = struct.pack('<Q', len(header)) + header + b'x'
        for suffix, contents in (('.data', safe), ('.bin', b'GGUF' + bytes(20)), ('.gguf', safe)):
            with self.subTest(suffix=suffix):
                model = self.directory / ('video' + suffix)
                encoder = self.directory / ('encoder' + suffix)
                model.write_bytes(contents); encoder.write_bytes(contents)
                args = self.request('--model', str(model), '--text-encoder', str(encoder))
                self.assertEqual(args.model, str(model.resolve()))
                self.assertEqual(args.text_encoder, encoder.resolve())
        for name, contents in (('bad.bin', b'not a tensor container'),
                               ('bad.onnx', safe)):
            path = self.directory / name
            path.write_bytes(contents)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.request('--model', str(path))
            with self.subTest(encoder=name), self.assertRaises(ValueError):
                self.request('--text-encoder', str(path))

    def test_native_video_components_round_trip_and_require_local_files(self):
        files = {}
        for name in ('model', 'vae', 'high', 'vision'):
            path = self.directory / (name + '.safetensors')
            path.write_bytes(b'fixture')
            files[name] = path
        args = self.request('--model', str(files['model']), '--video-vae', str(files['vae']),
                            '--high-noise-model', str(files['high']), '--vision-encoder', str(files['vision']),
                            '--moe-boundary', '0.81', '--high-noise-steps', '7', '--high-noise-guidance', '3.5')
        self.assertEqual((args.video_vae, args.high_noise_model, args.vision_encoder),
                         (files['vae'].resolve(), files['high'].resolve(), files['vision'].resolve()))
        self.assertEqual((args.moe_boundary, args.high_noise_steps, args.high_noise_guidance), (0.81, 7, 3.5))
        saved = self.document('video-components.json', configuration(args))
        restored = self.request('--config', str(saved))
        self.assertEqual((restored.video_vae, restored.high_noise_model, restored.vision_encoder),
                         (args.video_vae, args.high_noise_model, args.vision_encoder))
        self.assertEqual((restored.moe_boundary, restored.high_noise_steps, restored.high_noise_guidance),
                         (args.moe_boundary, args.high_noise_steps, args.high_noise_guidance))
        with self.assertRaisesRegex(ValueError, 'native.*file'):
            self.request('--video-vae', str(files['vae']))
        bad = self.directory / 'decoder.ONNX'; bad.write_bytes(b'fixture')
        for option in ('--video-vae', '--high-noise-model', '--vision-encoder'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.request('--model', str(files['model']), option, str(bad))
        for option, value in (('--moe-boundary', 'nan'), ('--moe-boundary', '1.1'),
                              ('--high-noise-steps', '0'), ('--high-noise-guidance', '-2')):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.request('--model', str(files['model']), '--high-noise-model', str(files['high']), option, value)
        deferred = self.request('--model', str(files['model']), '--moe-boundary', '0.8')
        self.assertIsNone(deferred.high_noise_model)
        self.assertEqual(deferred.moe_boundary, 0.8)

    def test_video_output_cannot_replace_a_component_shard(self):
        model = self.directory / 'model.safetensors'
        model.write_bytes(b'fixture')
        shard = self.directory / 'input.mp4'
        header = b'{"vae.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        shard.write_bytes(struct.pack('<Q', len(header)) + header + b'x')
        index = self.document('vae.safetensors.index.json', {'weight_map': {'vae.weight': shard.name}})
        with self.assertRaisesRegex(ValueError, 'must not replace'):
            self.request('--model', str(model), '--video-vae', str(index), '--output', str(shard), '--overwrite')

    def test_pytorch_shard_index_is_a_replayable_video_input_and_protects_shards(self):
        payload = pickle.dumps(0x1950a86a20f9469cfc6c, protocol=2) + b'fixture'
        first, second = self.directory / 'part.bin', self.directory / 'input.mp4'
        first.write_bytes(payload);second.write_bytes(payload)
        index = self.document('video-index.json', {'weight_map': {'first': first.name, 'second': second.name}})
        args = self.request('--model', str(index), '--text-encoder', str(index), '--output', str(self.directory/'movie.mp4'))
        self.assertEqual((args.model, args.text_encoder), (str(index), index))
        self.assertEqual(configuration(args)['model'], str(index))
        with self.assertRaisesRegex(ValueError, 'must not replace'):
            self.request('--model', str(index), '--output', str(second), '--overwrite')

    def test_defaults_make_a_five_second_temporal_generation_plan(self):
        args = self.request()
        self.assertEqual(args.max_frames, 120)
        self.assertEqual(args.shots[0]["sample_frames"], 65)
        self.assertEqual(args.shots[0]["ltx_frames"], 61)
        self.assertEqual(args.shots[0]["frames"], 120)
        self.assertEqual(args.fps, 24)
        self.assertEqual(args.animation_mode, "Video")
        self.assertIsNone(args.revision)
        self.assertTrue(args.local_files_only)
        self.assertEqual(args.shots[0]["camera"], ["none"])

    def test_low_vram_configuration_round_trips(self):
        args = self.request("--low-vram", "--no-cpu-text-encoding", "--no-vae-tiling")
        self.assertTrue(args.low_vram)
        self.assertEqual(args.offload, "sequential")
        self.assertTrue(args.cpu_text_encoding)
        self.assertTrue(args.vae_tiling)
        replay = self.document("low-vram-replay.json", configuration(args))
        restored = self.request("--config", str(replay))
        self.assertTrue(restored.low_vram)
        self.assertEqual(restored.offload, "sequential")
        self.assertTrue(restored.cpu_text_encoding and restored.vae_tiling)

    def test_duration_and_camera_mix_are_compiled_into_explicit_shots(self):
        args = self.request("--duration", "3", "--fps", "25", "--camera", "dolly-in", "pan-left")
        shot = args.shots[0]
        self.assertEqual((shot["frames"], shot["sample_frames"]), (75, 41))
        self.assertIn("forward", shot["effective_prompt"])
        self.assertIn("left", shot["effective_prompt"])
        self.assertEqual(len(CAMERA_MOTIONS), len(set(CAMERA_MOTIONS)))

    def test_invalid_parameters_fail_before_loading_runtime(self):
        cases = [("--duration", "nan"), ("--duration", "0"), ("--fps", "0"),
                 ("--frames", "1"), ("--frames", "-2"), ("--width", "321"),
                 ("--height", "0"), ("--steps", "0"), ("--guidance-scale", "nan"),
                 ("--seed", "-1"), ("--frames", "9", "--duration", "1"),
                 ("--camera", "static", "dolly-in"), ("--camera", "none", "pan-left"),
                 ("--camera", "pan-left", "pan-right"), ("--output", "bad.png"),
                 ("--device", "cpu", "--offload", "model"),
                 ("--model", "owner/model"), ("--revision", "main")]
        for tokens in cases:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.request(*tokens)

    def test_storyboard_has_exact_cut_positions_and_relative_image_paths(self):
        image = self.directory / "keyframe.png"
        image.write_bytes(b"image is decoded during media preflight")
        storyboard = self.document("shots.json", {"shots": [
            {"prompt": "A glass bottle on a table.", "frames": 9, "first_frame": "keyframe.png"},
            {"prompt": "The bottle turns.", "frames": 17, "continue_previous": True,
             "camera": ["orbit-left"]}]})
        args = self.request("--storyboard", str(storyboard), "--seed", "7")
        self.assertEqual(args.max_frames, 26)
        self.assertEqual([s["start_frame"] for s in args.shots], [0, 9])
        self.assertEqual([s["seed"] for s in args.shots], [7, 8])
        self.assertEqual(args.shots[0]["conditions"][0]["image"], str(image.resolve()))
        self.assertTrue(args.shots[1]["continue_previous"])

    def test_first_and_last_keyframes_condition_the_output_timeline(self):
        image = self.directory / "image.png"
        image.write_bytes(b"fixture")
        args = self.request("--frames", "16", "--first-frame", str(image), "--last-frame", str(image))
        self.assertEqual([c["frame"] for c in args.shots[0]["conditions"]], [0, 15])
        self.assertEqual(args.shots[0]["sample_frames"], 9)

    def test_storyboard_rejects_ambiguous_or_unusable_conditions(self):
        bad = [{"shots": []}, {"shots": [{"prompt": "x", "unknown": 1}]},
               {"shots": [{"prompt": "x", "continue_previous": True}]},
               {"shots": [{"prompt": "x", "frames": True}]},
               {"shots": [{"prompt": "x", "camera": "orbit-left"}]},
               {"shots": [{"prompt": "x", "conditions": [{"image": "missing.png", "frame": 0}]}]}]
        for data in bad:
            path = self.document("bad.json", data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.request("--storyboard", str(path))

    def test_config_cli_override_and_replay_preserve_plan(self):
        config = self.document("request.json", {"backend": "video", "prompt": "A river flows.",
                                                "duration": 2, "camera": ["pan-left"],
                                                "output": "river.mp4"})
        args = self.request("--config", str(config), "--prompt", "A waterfall flows.")
        self.assertEqual(args.prompt, "A waterfall flows.")
        self.assertEqual(args.output, self.directory / "river.mp4")
        replay = self.document("replay.json", configuration(args))
        self.assertEqual(args.shots, self.request("--config", str(replay)).shots)

    def test_frontdoor_video_and_json_routes_work_without_ml_imports(self):
        config = self.document("video.json", {"backend": "video", "frames": 9})
        for tokens in (["--backend", "video"], ["--config", str(config)]):
            result = subprocess.run([sys.executable, str(ROOT / "reference/generate.py"), "--model", MODEL,
                                     *tokens, "--print-config"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["backend"], "video")

    def test_output_cannot_destroy_its_storyboard(self):
        path = self.document("movie.json", {"shots": [{"frames": 9}]})
        with self.assertRaisesRegex(ValueError, "must not replace"):
            self.request("--storyboard", str(path), "--output", str(path.with_suffix(".mp4")), "--overwrite")


if __name__ == "__main__":
    unittest.main()
