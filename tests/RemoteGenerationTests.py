#!/usr/bin/env python3
"""Real loopback inference transport and local postprocessing, with no billable calls."""

from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import generate
import remote_generation
from remote_inference import RemoteInference
from generation_config import configuration_values
import video_options
from model_sources import ModelInput

spec = importlib.util.spec_from_file_location("remote_router_test", ROOT / "reference/generate.py")
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Pillow is required for decoded-media verification")
class RemoteGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="remote-generation-", dir=ROOT / "build")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.requests = []
        buffer = BytesIO()
        Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
        self.response = buffer.getvalue()
        self.content_type = "image/png"
        self.status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append({"body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                       "authorization": self.headers.get("Authorization")})
                self.send_response(owner.status)
                self.send_header("Content-Type", owner.content_type)
                self.send_header("Content-Length", str(len(owner.response)))
                if owner.status == 302:
                    self.send_header("Location", "/redirected")
                self.end_headers()
                self.wfile.write(owner.response)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.worker.start()
        self.addCleanup(self.close_server)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/infer"

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)

    def image_args(self, **values):
        _, args = generate.resolve_request({"model_api": self.endpoint, "width": 32, "height": 32,
                                           "output": str(self.directory / "image.png"), **values})
        return args

    def video_args(self, **values):
        return video_options.resolve_options(video_options.build_parser().parse_values({
            "model_api": self.endpoint, "model_family": "ltx", "width": 32, "height": 32,
            "frames": 9, "fps": 24, "output": str(self.directory / "video.mp4"), **values}))

    def make_video_response(self, frames=9):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("FFmpeg and FFprobe are required")
        path = self.directory / "response.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=32x32:rate=12", "-frames:v", str(frames), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(path)], capture_output=True, check=True, timeout=30)
        self.response = path.read_bytes()
        self.content_type = "video/mp4"

    def test_endpoint_image_uses_real_http_and_records_source_without_token(self):
        args = self.image_args(model_token_env="IILD_TEST_TOKEN", prompt="a red cube", seed=12)
        with patch.dict(os.environ, {"IILD_TEST_TOKEN": "secret-test-value"}):
            reports = remote_generation.generate_image(args)
        self.assertTrue(args.output.is_file())
        self.assertEqual(self.requests[0]["body"]["inputs"], "a red cube")
        self.assertEqual(self.requests[0]["body"]["parameters"]["seed"], 12)
        self.assertEqual(self.requests[0]["authorization"], "Bearer secret-test-value")
        self.assertEqual(reports[0]["model_input"]["kind"], "api")
        self.assertNotIn("secret-test-value", args.output.with_suffix(".json").read_text())
        self.assertTrue(reports[0]["image"]["verified_decode"])
        self.assertFalse(reports[0]["weights_verified_locally"])

    def test_remote_config_roundtrips_without_network_or_token_lookup(self):
        args = self.image_args(model_api=None, model_cloud="owner/model", model_provider="fal-ai")
        values = configuration_values(args)
        _, replay = generate.resolve_request(values)
        self.assertEqual(configuration_values(replay), values)
        self.assertEqual(self.requests, [])
        args = self.video_args()
        values = video_options.configuration(args)
        replay = video_options.resolve_options(video_options.build_parser().parse_values(values))
        remote_generation.validate_remote_video(replay)
        self.assertEqual(video_options.configuration(replay), values)

    def test_provider_model_id_is_sent_to_inference_client_without_weight_loading(self):
        args = self.image_args(model_api=None, model_cloud="owner/model", model_provider="fal-ai")
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface-hub is required for the provider adapter")
        with patch.dict(os.environ, {"HF_TOKEN": "cloud-test-token"}), patch.object(huggingface_hub, "InferenceClient") as client:
            client.return_value.text_to_image.return_value = Image.new("RGB", (32, 32), "blue")
            reports = remote_generation.generate_image(args)
        client.assert_called_once_with(model="owner/model", provider="fal-ai", token="cloud-test-token", timeout=300)
        self.assertEqual(client.return_value.text_to_image.call_args.kwargs["width"], 32)
        self.assertEqual(reports[0]["model_input"]["kind"], "cloud")
        self.assertEqual(self.requests, [])

    def test_cloud_video_uses_text_to_video_and_local_interpolator(self):
        self.make_video_response()
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface-hub is required for the provider adapter")
        args = self.video_args(model_api=None, model_cloud="owner/LTX", model_provider="fal-ai")
        with patch.dict(os.environ, {"HF_TOKEN": "cloud-test-token"}), patch.object(huggingface_hub, "InferenceClient") as client:
            client.return_value.text_to_video.return_value = self.response
            report = remote_generation.generate_video(args)
        self.assertEqual(client.return_value.text_to_video.call_args.kwargs["num_frames"], 9)
        self.assertEqual([stage["name"] for stage in report["stages"]], ["LTX", "Interpolator"])
        self.assertEqual(report["model_input"]["kind"], "cloud")
        self.assertEqual(self.requests, [])

    def test_http_failure_and_redirect_are_not_retried_and_do_not_publish(self):
        for status in (401, 500, 302):
            self.status = status
            before = len(self.requests)
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                remote_generation.generate_image(self.image_args())
            self.assertEqual(len(self.requests), before + 1)
            self.assertFalse((self.directory / "image.png").exists())
            self.assertFalse((self.directory / "image.json").exists())

    def test_malformed_response_preserves_existing_output(self):
        output = self.directory / "image.png"
        output.write_bytes(b"previous image")
        output.with_suffix(".json").write_bytes(b"previous report")
        self.response = b"not a PNG"
        with self.assertRaises(OSError):
            remote_generation.generate_image(self.image_args(overwrite=True))
        self.assertEqual(output.read_bytes(), b"previous image")
        self.assertEqual(output.with_suffix(".json").read_bytes(), b"previous report")

    def test_batch_cannot_overwrite_its_input_configuration(self):
        path = self.directory / "image-0001.json"
        path.write_text(json.dumps({"model_api": self.endpoint, "num_images": 2, "width": 32, "height": 32,
                                   "output": str(self.directory / "image.png"), "overwrite": True}))
        before = path.read_bytes()
        _, args = generate.resolve_arguments(generate.build_parser().parse_args(["--config", str(path)]))
        with self.assertRaisesRegex(ValueError, "configuration"):
            remote_generation.generate_image(args)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.requests, [])

    def test_mislabelled_image_format_is_rejected(self):
        buffer = BytesIO()
        Image.new("RGB", (32, 32), "red").save(buffer, format="JPEG")
        self.response = buffer.getvalue()
        with self.assertRaisesRegex(RuntimeError, "PNG image"):
            remote_generation.generate_image(self.image_args())
        self.assertFalse((self.directory / "image.png").exists())

    def test_unified_cli_executes_api_image_generation(self):
        output = self.directory / "from-cli.png"
        with redirect_stdout(StringIO()):
            self.assertEqual(router.main(["--model-api", self.endpoint, "--width", "32", "--height", "32",
                                          "--output", str(output)]), 0)
        self.assertTrue(output.is_file())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(json.loads(output.with_suffix(".json").read_text())["model_input"]["kind"], "api")

    def test_unsupported_local_controls_fail_before_network(self):
        for options in ({"lora": "adapter.safetensors"}, {"device": "cpu"}, {"animation_mode": "2D"}):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.image_args(**options)
        with self.assertRaisesRegex(ValueError, "Local execution"):
            remote_generation.generate_video(self.video_args(device="cpu"))
        self.assertEqual(self.requests, [])

    def test_ltx_declaration_is_required_for_remote_video(self):
        with self.assertRaisesRegex(ValueError, "model-family ltx"):
            self.video_args(model_family=None)

    def test_endpoint_video_gets_interpolated_to_the_requested_final_timeline(self):
        self.make_video_response()
        report = remote_generation.generate_video(self.video_args())
        self.assertEqual(len(self.requests), 1)
        params = self.requests[0]["body"]["parameters"]
        self.assertEqual((params["num_frames"], params["frame_rate"]), (9, 12))
        self.assertEqual([stage["name"] for stage in report["stages"]], ["LTX", "Interpolator"])
        self.assertEqual((len(report["source_frames"]), len(report["frames"])), (5, 9))
        self.assertTrue(report["interpolation"]["anchors_preserved"])
        self.assertTrue(report["video"]["verified_decode"])
        self.assertEqual(report["video"]["fps"], 24)

    def test_explicit_low_fps_ltx_skips_the_second_stage(self):
        self.make_video_response()
        report = remote_generation.generate_video(self.video_args(fps=8))
        self.assertEqual([stage["name"] for stage in report["stages"]], ["LTX"])
        self.assertFalse(report["interpolation"]["enabled"])
        self.assertEqual(report["video"]["fps"], 8)

    def test_wrong_video_frame_count_never_publishes(self):
        self.make_video_response(frames=8)
        args = self.video_args()
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            remote_generation.generate_video(args)
        self.assertFalse(args.output.exists())
        self.assertFalse(args.output.with_suffix(".json").exists())
        self.assertFalse(args.output.with_name(args.output.stem + "-frames").exists())

    def test_unified_cli_dispatches_both_remote_kinds_and_keeps_animation_policy(self):
        for flags in (("--model-api", self.endpoint),
                      ("--model-cloud", "owner/model", "--model-provider", "fal-ai")):
            with self.subTest(flags=flags), redirect_stdout(StringIO()) as stream:
                self.assertEqual(router.main([*flags, "--print-config"]), 0)
                self.assertTrue(json.loads(stream.getvalue()))
            with self.assertRaises(SystemExit):
                router.main([*flags, "--fps", "8", "--print-config"])
            with self.assertRaises(SystemExit):
                router.main([*flags, "--output", str(self.directory / "video.gif"), "--print-config"])
        with redirect_stdout(StringIO()) as stream:
            self.assertEqual(router.main(["--model-api", self.endpoint, "--model-family", "ltx",
                                          "--output", str(self.directory / "video.mp4"), "--print-config"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["fps"], 24)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
