"""Foreground preparation belongs to the SDK and must never generate an image."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference/diffusers"))
import inference_session
import inference_worker


class ForegroundInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="foreground-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.model = self.directory / "model.safetensors"
        self.model.write_bytes(b"weights")

    def serve(self, requests, generate):
        stream = io.BytesIO(("\n".join(json.dumps({"schema": "iild-worker-request-v1", "id": str(i), **item})
                                      for i, item in enumerate(requests)) + "\n").encode())
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertEqual(inference_worker.serve(generate, stream), 0)
        lines = output.getvalue().splitlines()
        ready = json.loads(next(line[11:] for line in lines if line.startswith("IILD_READY ")))
        results = [json.loads(line[12:]) for line in lines if line.startswith("IILD_RESULT ")]
        return ready, results

    def test_foreground_places_once_then_waits_without_generation_or_output(self):
        loader, place = Mock(return_value=object()), Mock(return_value=object())
        calls = []
        def generate(arguments):
            calls.append("prepare" if inference_session.is_preparing() else "generate")
            inference_session.cached_pipeline("model", [self.model], loader)
            def placement(previous):
                inference_session.record_device_placement()
                return place()
            inference_session.cached_placement("mps", placement)
            inference_session.record_execution("mps", "none", str(self.model))
        ready, results = self.serve([
            {"action": "foreground", "foreground": True, "arguments": ["model"]},
            {"arguments": ["model", "new prompt"]},
            {"action": "foreground", "foreground": False},
            {"action": "foreground", "foreground": True, "arguments": ["model"]},
            {"action": "foreground", "foreground": True},
        ], generate)
        self.assertIn("foreground-residency", ready["capabilities"])
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual(calls, ["prepare", "generate", "prepare"])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(place.call_count, 1)
        self.assertEqual(results[0]["residency"]["state"], "ready")
        self.assertTrue(results[0]["residency"]["gpu_resident"])
        self.assertFalse(results[2]["residency"]["foreground"])
        self.assertFalse(results[-1]["residency"]["ready"])
        self.assertEqual(results[1]["cache"]["device_placements"], 0)
        self.assertEqual(list(self.directory.iterdir()), [self.model])

    def test_invalid_lifecycle_never_dispatches_generation(self):
        generate = Mock()
        _, results = self.serve([
            {"action": "foreground", "foreground": "true", "arguments": ["model"]},
            {"action": "foreground", "foreground": False, "arguments": ["model"]},
            {"action": "unknown", "arguments": ["model"]},
        ], generate)
        self.assertTrue(all(not result["ok"] for result in results))
        generate.assert_not_called()

    def test_no_model_waits_and_failed_preparation_does_not_claim_readiness(self):
        generate = Mock(return_value=0)
        _, results = self.serve([
            {"action": "foreground", "foreground": True},
            {"action": "foreground", "foreground": True, "arguments": ["unsupported"]},
        ], generate)
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0]["residency"]["state"], "waiting-model")
        self.assertFalse(results[1]["ok"])
        self.assertFalse(results[1]["residency"]["ready"])
        self.assertFalse(inference_session.is_preparing())

    def test_direct_session_failure_clears_the_previously_ready_model(self):
        def prepare():
            inference_session.cached_pipeline("model", [self.model], object)
            inference_session.record_execution("mps", "none", str(self.model))
        with inference_session.InferenceSession() as session:
            session.set_foreground(True, prepare)
            self.assertTrue(session.residency()["ready"])
            with self.assertRaisesRegex(ValueError, "new model failed"):
                session.set_foreground(True, Mock(side_effect=ValueError("new model failed")))
            self.assertFalse(session.residency()["ready"])
            self.assertFalse(inference_session.is_preparing())
            session.set_foreground(True, prepare)
            self.assertEqual(session.set_foreground(True, lambda: 2), 2)
            self.assertFalse(session.residency()["ready"])
            session.set_foreground(True, prepare)
            self.assertTrue(session.residency()["gpu_resident"])

    def test_prepare_only_standalone_bypasses_image_output_and_request_files(self):
        import standalone_image
        from types import SimpleNamespace
        args = SimpleNamespace(print_config=False, validate_only=False, output_dir=self.directory / "output",
                               output_was_default=True, work_dir=self.directory / "work")
        with (inference_session.InferenceSession() as session,
              patch.object(standalone_image, "resolve_arguments", return_value=(object(), args)),
              patch.object(standalone_image, "build_parser") as parser,
              patch.object(standalone_image.generate, "run", return_value=0) as run):
            session.preparing = True
            self.assertEqual(standalone_image.main([]), 0)
            run.assert_called_once()
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.work_dir.exists())


if __name__ == "__main__":
    unittest.main()
