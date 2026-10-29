#!/usr/bin/env python3
"""The worker must retain actionable errors when a CLI exits with a code."""
import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import inference_worker


class InferenceWorkerFailureTests(unittest.TestCase):
    def run_requests(self, names, generate):
        requests = [dict(schema="iild-worker-request-v1", id=str(index), arguments=[name])
                    for index, name in enumerate(names)]
        incoming = io.BytesIO(("\n".join(map(json.dumps, requests)) + "\n").encode())
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", errors):
            self.assertEqual(inference_worker.serve(generate, incoming), 0)
            self.assertIs(sys.stderr, errors)
        results = [json.loads(line[12:]) for line in output.getvalue().splitlines()
                   if line.startswith("IILD_RESULT ")]
        return results, errors.getvalue()

    def test_numeric_exit_keeps_cli_error_and_streams_it(self):
        message = "Video generation failed: LTX weight preparation failed."
        def generate(arguments):
            if arguments[0] == "failure":
                argparse.ArgumentParser().exit(2, message + "\n")
            return 0
        results, stderr = self.run_requests(["failure", "success"], generate)
        self.assertEqual(stderr, message + "\n")
        self.assertEqual(results[0]["exit_code"], 2)
        self.assertEqual(results[0]["error"], message)
        self.assertFalse(results[0]["ok"])
        self.assertTrue(results[1]["ok"])
        self.assertEqual(results[1]["error"], "")

    def test_real_video_entrypoint_returns_the_native_failure(self):
        import generate_video
        args = SimpleNamespace(list_camera_motions=False, model_input=SimpleNamespace(kind="local"),
                               print_config=False, model="fixture.safetensors")
        parser = Mock()
        parser.parse_args.return_value = args
        parser.exit.side_effect = lambda code, message: argparse.ArgumentParser().exit(code, message)
        native = SimpleNamespace(generate_video=Mock(side_effect=RuntimeError("segment 4 weight preparation failed")))
        with (patch.object(generate_video, "build_parser", return_value=parser),
              patch.object(generate_video, "resolve_options", return_value=args),
              patch.object(Path, "is_file", return_value=True),
              patch.dict(sys.modules, {"native_video": native})):
            results, stderr = self.run_requests(["failure"], generate_video.main)
        self.assertEqual(results[0]["exit_code"], 2)
        self.assertIn("segment 4 weight preparation failed", results[0]["error"])
        self.assertIn(results[0]["error"], stderr)

    def test_numeric_exit_without_message_is_explicit_and_isolated(self):
        def generate(arguments):
            if arguments[0] == "warning":
                print("unrelated warning", file=sys.stderr)
                return 0
            raise SystemExit(2)
        results, stderr = self.run_requests(["warning", "failure"], generate)
        self.assertIn("unrelated warning", stderr)
        self.assertNotIn("unrelated warning", results[1]["error"])
        self.assertIn("2", results[1]["error"])
        self.assertNotEqual(results[1]["error"], "2")

    def test_string_exit_remains_unchanged(self):
        def generate(_):
            raise SystemExit("direct failure")
        results, _ = self.run_requests(["failure"], generate)
        self.assertEqual(results[0]["error"], "direct failure")
        self.assertEqual(results[0]["exit_code"], 1)

    def test_foreground_preparation_failure_preserves_its_message(self):
        requests = [
            dict(schema="iild-worker-request-v1", id="prepare", action="foreground",
                 foreground=True, arguments=["failure"]),
            dict(schema="iild-worker-request-v1", id="next", arguments=["success"]),
        ]
        incoming = io.BytesIO(("\n".join(map(json.dumps, requests)) + "\n").encode())
        output, errors = io.StringIO(), io.StringIO()
        def generate(arguments):
            if arguments == ["failure"]:
                argparse.ArgumentParser().exit(2, "Could not prepare the video model.\n")
            return 0
        with patch("sys.stdout", output), patch("sys.stderr", errors):
            self.assertEqual(inference_worker.serve(generate, incoming), 0)
        results = [json.loads(line[12:]) for line in output.getvalue().splitlines()
                   if line.startswith("IILD_RESULT ")]
        self.assertEqual(results[0]["error"], "Could not prepare the video model.")
        self.assertTrue(results[1]["ok"])
        self.assertEqual(results[1]["error"], "")

    def test_captured_error_is_bounded_and_keeps_the_end(self):
        def generate(_):
            argparse.ArgumentParser().exit(2, "x" * 5000 + " END\n")
        results, stderr = self.run_requests(["failure"], generate)
        self.assertEqual(len(stderr), 5005)
        self.assertLessEqual(len(results[0]["error"]), 4000)
        self.assertTrue(results[0]["error"].endswith(" END"))


if __name__ == "__main__":
    unittest.main()
